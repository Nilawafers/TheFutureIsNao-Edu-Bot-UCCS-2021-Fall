import qi
import almath
import argparse

def main(session):
    """
    A test script for moving the Nao robot using the Cartesian
    Control API in the NAOqi SDK
    """
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")
    speech = session.service("ALTextToSpeech")

    # initialize stance
    log("waking")
    motion.wakeUp()

    goto_stand(motion, posture)

    log("preparing to move")
    motion.moveInit()
    # print(motion.getPosition("Torso", 2, True))

    # log("crouching")
    # posture.goToPosture("Crouch", 0.5)

    # **********************************************************
    # not strictly necessary, but make subsequent motion smoother
    # **********************************************************
    motion.setStiffnesses("RArm", 0.0)
    motion.setStiffnesses("LArm", 0.0)
    motion.setStiffnesses("Head", 0.0)
    
    # open the left hand
    log("opening hand")
    motion.openHand("LHand")
    motion.closeHand("LHand")

    

    # move the left arm
    motion.wbEnableEffectorControl("LArm", True)
    motion.wbEnableEffectorControl("RArm", True)

    lpoints = [
	[0.15457427501678467, 0.131572425365448, 0.5019456148147583,
            -1.5897225141525269, -0.935154914855957, 0.20292489230632782],
	[0.15036171674728394, 0.10574427992105484, 0.5386977791786194,
            -1.668771505355835, -1.0586074590682983, 0.09279955923557281],
	[0.16299599409103394, 0.0846848413348198, 0.5012325048446655,
            -1.6748456954956055, -0.8622304797172546, -0.027583902701735497],
	[0.15538376569747925, 0.13587330281734467, 0.5069384574890137,
            -1.7119406461715698, -0.9142926931381226, 0.27707982063293457]
    ]

    # rpoints = list(lpoints)
    # for point in rpoints:
	 #point[1] *= -1

    motion.setStiffnesses("LArm", 1.0)
    
    motion.positionInterpolations(["LArm"], [2], [lpoints], [7], [[3, 6, 9, 12]])
    # for point in points:
	# print(point)
	# motion.positionInterpolations("LArm", 2, point, 7, 3)
    # motion.positionInterpolations(["LArm", "RArm"], [2, 2], [lpoints, rpoints], [7, 7], [[3, 6, 9, 12], [3, 6, 9, 12]])

    # for point in points:
	# point[1] *= -1

    # print(points)

    # motion.setStiffnesses("RArm", 1.0)
    # motion.positionInterpolations("RArm", 2, rpoints, 7, [3, 6, 9, 12])
    
    # first_move(motion)
    # second_move(motion)
    # third_move(motion)

    # log("resting")
    # motion.rest()


def first_move(motion):
    currPos = motion.getPosition("LArm", 2, True)
    currPos[0] += 0.04
    currPos[2] += 0.12
    # log("closing hand")
    # motion.closeHand("LHand")

    execute_move(motion, currPos, 1)


def second_move(motion):
    currPos = motion.getPosition("LArm", 2, True)
    currPos[1] -= 0.08
    currPos[0] += 0.04

    execute_move(motion, currPos, 2)


def third_move(motion):
    currPos = motion.getPosition("LArm", 2, True)
    currPos[1] += 0.16

    execute_move(motion, currPos, 3)
    

def execute_move(motion, moveTo, count):
    motion.setStiffnesses("LArm", 1.0)
    log("moving arm to [" + str(count) + "]")
    motion.positionInterpolations("LArm", 2, moveTo, 7, 3)
    motion.openHand("LHand")
    motion.closeHand("LHand")


def goto_stand(motion, posture):
    # **********************************************************
    # setting stiffnesses before going to StandInit puts the
    # robot in a slightly different starting position; however,
    # subsequent motion still works without doing so
    # **********************************************************
    motion.setStiffnesses("Body", 1.0)
    motion.setStiffnesses("LArm", 0.0)
    motion.setStiffnesses("RArm", 0.0)
    log("standing")
    posture.goToPosture("StandInit", 0.5)


def log(info):
    print ("[INFO] " + str(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.42.0.60",
            help="Robot IP address.")
    args = parser.parse_args()

    session = qi.Session()

    try:
        session.connect("tcp://"+ str(args.ip) +":9559")
        log("Connected to robot at: " + str(args.ip))
    except RuntimeError:
        print("Can't connect to Nao at ip: " + str(args.ip))
        sys.exit(1)
    main(session)
