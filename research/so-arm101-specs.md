# SO-ARM101 Deep Dive

## Overview
The SO-ARM101 is the current-generation low-cost, open-source 6-DOF robotic arm designed by TheRobotStudio in collaboration with Hugging Face. It is the standard hardware for the LeRobot ecosystem, superseding the SO-ARM100.

## Key Improvements over SO-100
- **No gear removal needed** for leader arm assembly (SO-100 required this)
- **Improved wiring** — simpler and cleaner cable routing
- **Different gear ratios** on leader arm motors for easier manual operation
- **Better assembly instructions** and community support

## Motor Details: Feetech STS3215
- **Type:** Serial bus servo (half-duplex TTL)
- **Protocol:** Feetech SCS/STS serial protocol
- **Resolution:** 4096 positions per revolution (0.088°)
- **Voltage options:** 7.4V or 12V
- **Stall torque:** 16.5 kg·cm @ 6V (7.4V version), 30 kg·cm (12V version)
- **Speed:** ~0.16 sec/60° (no load)
- **Feedback:** Position, speed, load, voltage, temperature
- **Communication:** 1Mbps UART, daisy-chainable

### SO-101 Gear Ratios
- **Joints 1 (shoulder_pan):** 1/345 gear ratio (STS3215 C001)
- **Joints 2-4:** Mix of 1/345 and 1/191 gear ratios
- **Joints 5-6:** 1/147 gear ratio (C046)
- **Leader arm:** Different gear ratios for easier manual operation

## Controller Board
- **Waveshare Bus Servo Adapter** (~$10)
- USB-C connection to computer
- Powers and communicates with all 6 servos via daisy chain
- Jumper setting: Channel B (USB mode)
- Separate 5V power supply for servos

## Serial Protocol Key Registers
| Register | Address | Size | Description |
|----------|---------|------|-------------|
| Goal Position | 42 | 2 bytes | Target position (0-4095) |
| Present Position | 56 | 2 bytes | Current position |
| Present Speed | 58 | 2 bytes | Current speed |
| Present Load | 60 | 2 bytes | Current load |
| Present Voltage | 62 | 1 byte | Current voltage |
| Present Temperature | 63 | 1 byte | Current temp |
| Torque Enable | 40 | 1 byte | 0=off, 1=on |
| Moving Speed | 46 | 2 bytes | Max speed limit |
| ID | 5 | 1 byte | Servo ID (1-253) |
| Baudrate | 6 | 1 byte | Communication speed |

## Python Control
```python
# Using scservo_sdk (feetech SDK)
from scservo_sdk import *

port = PortHandler('/dev/ttyACM0')
packet = PacketHandler(0)
port.openPort()
port.setBaudRate(1000000)

# Enable torque
packet.write1ByteTxRx(port, servo_id, 40, 1)

# Set position (0-4095, center=2048)
packet.write2ByteTxRx(port, servo_id, 42, 2048)

# Read current position
pos, result, error = packet.read2ByteTxRx(port, servo_id, 56)

# Set speed limit (lower = slower, smoother)
packet.write2ByteTxRx(port, servo_id, 46, 200)
```

## Joint Workspace (approximate)
| Joint | Range | Notes |
|-------|-------|-------|
| Shoulder Pan | ~270° | Base rotation |
| Shoulder Lift | ~180° | Forward/backward |
| Elbow Flex | ~180° | Bend |
| Wrist Flex | ~180° | Up/down |
| Wrist Roll | ~360° | Continuous-ish |
| Gripper | ~90° | Open/close |
