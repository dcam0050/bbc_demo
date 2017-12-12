#!/usr/bin/env bash
yarp connect /face/eyelids /icubSim/face/eyelids &
yarp connect /face/image/out /icubSim/texture/face &
yarp connect /icub/face/emotions/out /icubSim/face/raw/in 