# Robot arm missions

This is the group code for the 4-joint robot arm. It can check kinematics on your computer, draw a square with an X, visit points around a circle, and follow a ring on the table with the wrist camera. The arm itself is optional until you are ready to move it.

Positions are in millimetres. Angles are in degrees.

## 1. Install Git and Python

You need Git and Python 3.10 or newer. On Ubuntu:

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip
```

Check that Python is there:

```bash
python3 --version
```

## 2. Download the code

On the repository page, click the green **Code** button and copy the HTTPS address. Then:

```bash
git clone PASTE_THE_ADDRESS_HERE
cd dtu-robotics
```

## 3. Install the Python packages

Do this once, from the `dtu-robotics` folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r assignment/requirements.txt
```

`source venv/bin/activate` is required in every new terminal before the later commands. Your prompt should start with `(venv)`.

The camera window also needs `ffplay`, which comes with FFmpeg:

```bash
sudo apt install ffmpeg
```

## 4. Try it without the robot

Stay in `dtu-robotics` and go into the assignment folder. These commands only print numbers. They do not open the USB port.

```bash
cd assignment
```

Where the hand would be if every joint is at 0°:

```bash
python MyRobot.py --check
```

The joint angles that reach a chosen point. Here the point is x=150 mm, y=0, z=120 mm, pitch=0°:

```bash
python MyRobot.py --pose 150 0 120 0
```

The square-and-X drawing, as a list of moves:

```bash
python MyRobot.py --check --mission square_with_x
```

The circle of 37 points:

```bash
python MyRobot.py --check --mission problem_3
```

The same path as two drawings, with the joint angles under them. The left plot is the top view. The right plot is the arm seen from the side. This does not connect to the arm. Left and right step through the commands. Space plays them.

```bash
python MyRobot.py --view --mission problem_3
```

A mission that picks its target when it runs, then shows the arm that command would produce. The first pose is fixed. The second is `--pose`, or a built-in stand-in if you leave `--pose` out. This does not connect to the arm. The same window as above: left and right step, space plays.

```bash
python MyRobot.py --check --mission dynamic
python MyRobot.py --view --mission dynamic --pose 150 40 150 0
```

Ring following with no arm and no camera. A stand-in ring sits 30 mm off the point the camera is looking at. The printout is one 18 mm slide, then the short steps that bring that ring to the image center. The tool stays at the look height, about 127 mm, and the pitch stays at −90°.

```bash
python MyRobot.py --check --mission point_ring
python MyRobot.py --view --mission point_ring
```

One pose, or the zero pose if no angles are given:

```bash
python MyRobot.py --view --pose 150 0 120 0
python MyRobot.py --view
```

`--view` needs matplotlib. If it is missing: `pip install matplotlib`

Problem 4 prints the Jacobian matrices at four points on that circle. It does not move the arm:

```bash
python -m missions.problem_4
```

If a mission name is wrong, the program prints the names it knows.

## 5. Plug in the arm

1. Power the arm.
2. Plug the USB cable into the computer.
3. The code looks for the arm at `/dev/ttyACM0`.

Your user account needs permission to use that port. On Ubuntu, run this once, then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

If the arm is on a different port, change `DEVICENAME` near the top of `assignment/robot/bus.py`.

## 6. Run a mission on the arm

Clear the space around the arm. The first move goes from wherever the arm is sitting to the first point in the mission.

From the `assignment` folder, with `(venv)` active:

```bash
python MyRobot.py --mission square_with_x
```

That draws a 50 mm square and then an X inside it. The pen height is z=180 mm and the pitch is 60°. To move or resize it, edit the numbers at the top of `assignment/missions/square_with_x.py`, save the file, and run the same command again. Preview the new path first with `--check`.

The circle mission visits the center `(150, 0, 120)` and then 37 points on a circle of radius 32 mm in the yz plane:

```bash
python MyRobot.py --mission problem_3
```

When the mission finishes, the motors turn off and you can move the arm by hand again.

A pose check for when the arm does not match the model. It holds nine poses for 5 seconds each and prints the position the model expects. The joint commands stay inside the rotation limits. Joint 1 only yaws to ±125°, five degrees short of its ±130° stop, so the forearm never points straight back. The last two poses ask for a point and let inverse choose the joints. Preview it before running it on the arm:

```bash
python MyRobot.py --check --mission known_poses
python MyRobot.py --view --mission known_poses
python MyRobot.py --mission known_poses
```

Follow a ring on the table. The arm stays above the ring. Put one ring on the table in front of the base, where the camera can see it from above. The wrist camera is `/dev/video4`. Pass `--camera /dev/videoN` if it is on another device.

The arm first goes to the look pose, joints `(0, −20, 85, 115)`, and holds there for 2 seconds. The camera points straight down. Pitch stays at −90° after that, and the tool stays at the height of this pose, about 127 mm, for the slide and every step that follows.

Once the ring sits still in the picture, the camera slides 18 mm sideways at that same height. The shift of the ring in the image is the scale, in pixels per millimetre. The arm then creeps toward the ring, at most 2 mm per command, while the camera window stays open. It holds when the ring is within 30 px of the image center, and starts again if the ring drifts out past 50 px. Move the ring and the arm follows.

`q` in the camera window, or closing it, stops the loop and leaves the arm where it is. The motors turn off when the mission ends. The yellow circle is the ring. The cyan cross is the image center, and the cyan line is the pixel error from that cross to the hole.

```bash
python MyRobot.py --mission point_ring
python MyRobot.py --mission point_ring --camera /dev/video4
```

`--pose X Y Z PITCH` still starts at the look pose, then goes to that point once and skips the camera. The pitch it sends is −90°.

```bash
python MyRobot.py --mission point_ring --pose 180 0 127 -90
```

To check the ring detector alone, with no arm, see section 7.

## 7. Record poses by hand

With the arm plugged in, from the `assignment` folder:

```bash
python MyRobot.py --no-camera
```

Torque stays off. Move the joints by hand. The terminal shows the live angles.

| Key | What it does |
| --- | --- |
| `r` | Save the pose the arm is in right now |
| `u` | Delete the last saved pose |
| `c` | Start the min/max display over from the current pose |
| `q` | Save and quit |

Play those poses back later:

```bash
python MyRobot.py --play --no-camera
```

The saved files are `assignment/teach_history.json` and `assignment/teach_limits.json`. They stay on your computer and are not part of the git download.

To also open a camera window, leave off `--no-camera`:

```bash
python MyRobot.py
```

Search the webcam for a ring, with no arm connection. The yellow circle is the ring. The cyan cross is the stylus pixel, currently the image center, and the cyan line is the pixel error from that cross to the hole: positive `du` is right, positive `dv` is down. Press `q` to close it. `--camera /dev/videoN` picks a different device. The ring mission uses this same detector on the wrist camera.

```bash
python MyRobot.py --vision
python MyRobot.py --vision --camera /dev/video4
```

## If something goes wrong

**`Failed to open the port`**  
The USB cable is unplugged, the arm has no power, or the port name is not `/dev/ttyACM0`. List serial ports with `ls /dev/ttyACM* /dev/ttyUSB*`.

**`Permission denied` on `/dev/ttyACM0`**  
Run the `dialout` command in section 5, then log out and back in.

**`No module named ...`**  
The terminal is not in the `assignment` folder, or `(venv)` is missing. Run `source venv/bin/activate` from the `dtu-robotics` folder, then `cd assignment`.

**`ffplay is not installed`**  
Run `sudo apt install ffmpeg`, or add `--no-camera` so the program skips the camera window.
