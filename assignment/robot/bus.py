import time

from dynamixel_sdk import PacketHandler, PortHandler


class DynamixelBus:
    """Serial link to the AX-12A servos. Speaks in register reads and writes."""

    ADDR_TORQUE_ENABLE = 24
    ADDR_GOAL_POSITION = 30
    ADDR_MOVING_SPEED = 32
    ADDR_TORQUE_LIMIT = 34
    ADDR_PRESENT_POSITION = 36

    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000
    DEVICENAME = "/dev/ttyACM0"

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    COMM_SUCCESS = 0
    TICKS_PER_DEG = 1.0 / 0.29
    DEG_PER_TICK = 0.29

    # -1000 is COMM_PORT_BUSY: a previous packet left the port lock set.
    # -3001/-3002 are a missed or garbled status packet.
    _RETRY_RESULTS = (-1000, -1001, -1002, -3001, -3002)

    def __init__(self, device_name=None):
        self.device_name = device_name or self.DEVICENAME
        self.port_handler = None
        self.packet_handler = None

    def open(self):
        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        if not self.port_handler.openPort():
            print("Failed to open the port! Reconnect Robot.")
            raise RuntimeError("openPort failed")
        print("Succeeded to open the port!")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            print("Failed to change the baudrate! Reconnect Robot.")
            raise RuntimeError("setBaudRate failed")
        print("Succeeded to change the baudrate!")
        # The SDK's flush waits for the USB chip to drain before every packet.
        # On /dev/ttyACM* that often raises Linux EIO after a few commands.
        self.port_handler.clearPort = self._clear_rx_buffer

    def close(self):
        if self.port_handler is not None:
            self.port_handler.closePort()
        print("Port closed.")

    @classmethod
    def deg_to_ticks(cls, deg):
        return int(round(deg * cls.TICKS_PER_DEG))

    @classmethod
    def ticks_to_deg(cls, ticks):
        return ticks * cls.DEG_PER_TICK

    def _clear_rx_buffer(self):
        try:
            if self.port_handler.ser.in_waiting:
                self.port_handler.ser.reset_input_buffer()
        except Exception:
            pass

    def _linux_io_error(self, exc):
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 5:
            return True
        return "Input/output error" in str(exc)

    def _reopen_port(self):
        try:
            if self.port_handler.is_open:
                self.port_handler.closePort()
        except Exception:
            self.port_handler.is_open = False
        time.sleep(0.3)
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to reopen {self.device_name}")
        self.port_handler.is_using = False
        self.port_handler.clearPort = self._clear_rx_buffer

    def _transact(self, operation, what):
        last_error = None
        for attempt in range(5):
            self.port_handler.is_using = False
            try:
                outcome = operation()
            except Exception as e:
                self.port_handler.is_using = False
                last_error = e
                if self._linux_io_error(e):
                    try:
                        self._reopen_port()
                    except Exception as reopen_error:
                        last_error = reopen_error
                time.sleep(0.05)
                continue
            self.port_handler.is_using = False

            if len(outcome) == 3:
                value, result, error = outcome
            else:
                value = None
                result, error = outcome

            if result == self.COMM_SUCCESS and error == 0:
                time.sleep(0.01)
                return value

            last_error = IOError(f"{what} result={result}, error={error}")
            if result in self._RETRY_RESULTS and attempt < 4:
                time.sleep(0.05)
                continue
            raise last_error
        raise last_error

    def write1(self, dxl_id, addr, value):
        self._transact(
            lambda: self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, int(value)),
            f"DXL write1 error id={dxl_id}, addr={addr},",
        )

    def write2(self, dxl_id, addr, value):
        self._transact(
            lambda: self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, int(value)),
            f"DXL write2 error id={dxl_id}, addr={addr},",
        )

    def read2(self, dxl_id, addr):
        return self._transact(
            lambda: self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr),
            f"DXL read2 error id={dxl_id}, addr={addr},",
        )
