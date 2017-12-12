#!/usr/bin/env bash
yarpdev --device speech --language en-GB --pitch 90 --speed 110 &
p1=echo $!
iSpeak --package speech-dev &
p2=echo $!
sleep 2
yarp connect /iSpeak/speech-dev/rpc /icub/speech:rpc
yarp write /writer /iSpeak
kill $p1
kill $p2