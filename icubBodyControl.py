#!/usr/bin/env python

import sys
import time
import readline
import warnings
import numpy as np
import yarp
import os
from os.path import join
import subprocess
import copy
import signal
from xml.etree import ElementTree as ET
from operator import add
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class ICubBodyControl(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)
        self.interrupted = False
        self.portsList = dict()
        self.controlPorts = dict()
        self.monitorPorts = dict()
        self.controlBottles = dict()
        self.rpcPort = None
        self.rpcPort = None
        self.windowed = True
        self.persistence = True
        self.terminal = 'xterm'
        self.robot_name = 'icub'
        self.gestures_dict = dict()
        self.all_gestures_list = []
        self.all_gestures_files_list = []
        self.ctp_processes = []

        self.parts = ['head', 'left_arm', 'right_arm', 'torso']
        self.parts_processes = []
        self.root_gesture_dir = "/home/icub/user_files/icub_gestures"

    def configure(self, rf):
        persistence_val = rf.find('persistence').toString_c().lower()
        windowed_val = rf.find('windowed').toString_c().lower()
        robot_name_val = rf.find('robot').toString_c()

        if persistence_val != '':
            self.persistence = persistence_val == 'true'

        if windowed_val != '':
            self.windowed = windowed_val == 'true'

        if robot_name_val != '':
            self.robot_name = robot_name_val

        # Setting up rpc port
        self.portsList['rpc'] = yarp.Port()
        self.portsList['rpc'].open("/body_control/rpc:i")
        self.attach(self.portsList['rpc'])

        for part in self.parts:
            # Assemble command
            cmd_args = ['ctpService', '--robot', self.robot_name, '--part', part]

            # Start ctpService process and store pid
            ret = self.start_process(' '.join(cmd_args))

            # Connect to ctpService process
            if ret is not None:
                self.ctp_processes.append(ret)
                portName_control = "/body_control/" + part + "/rpc:o"
                self.controlPorts[part] = yarp.RpcClient()
                self.controlPorts[part].open(portName_control)
                self.controlBottles[part] = yarp.Bottle()

                portName_monitor = "/body_control/" + part + "/monitor:i"
                self.monitorPorts[part] = yarp.BufferedPortBottle()
                self.monitorPorts[part].open(portName_monitor)

                print "/" + join(self.robot_name, part, "state:o"), join("/ctpservice", part, "rpc")
                time.sleep(0.5)
                yarp.Network.connect(portName_control,  join("/ctpservice", part, "rpc"))
                yarp.Network.connect("/" + join(self.robot_name, part, "state:o"), portName_monitor, "udp")
            else:
                print "Error setting up ctpServices"
                return False

        # Load gestures from xml files into dictionary
        self.load_gestures()
        print "loaded gestures"

        # te = yarp.Bottle()
        # te.addString("move")
        # te.addString("left_arm_kill")
        # te.addString("mirror")
        # te.addString("delay=1.8")
        #
        # self.do_action("left_arm_kill", args=te)

        return True

    def close(self):
        print('Exiting ...')
        time.sleep(2)

        for j in self.portsList.keys():
            self.close_port(self.portsList[j])

        for j in self.controlPorts.keys():
            self.close_port(self.controlPorts[j])

        for p in self.ctp_processes:
            p.send_signal(signal.SIGINT)
            p.wait()

        return True

    def start_process(self, cmd):
        if self.persistence:
            command = "bash -c \"" + cmd + "; exec bash\""
        else:
            command = "bash -c \"" + cmd + "\""

        print('cmd: ' + str(cmd))

        c = None
        if self.windowed:
            c = subprocess.Popen([self.terminal, '-e', command], shell=False)
        else:
            c = subprocess.Popen([cmd], shell=True)

        return c

    @staticmethod
    def check_add(dict_add, key_add, type_add=None):
        if isinstance(dict_add, dict):
            if key_add not in dict_add.keys():
                dict_add[key_add] = type_add

    def send_ctp_messages(self, msg_list):
        # Send all ctp commands to ctpService queue
        rep_bot = yarp.Bottle()
        for part in msg_list:
            curr_bot = self.controlBottles[part]
            for ts in msg_list[part]:
                curr_bot.clear()
                curr_bot.fromString(ts[1])
                self.controlPorts[part].write(curr_bot, rep_bot)

    @staticmethod
    def construct_ctp_message(duration, positions, offset=None):
        msg = []
        time_total = 0
        for n in range(len(duration)):
            time_total += duration[n]
            if offset is not None:
                offset_pos = map(add, positions[n], offset)
            else:
                offset_pos = positions[n]
            m = ['[ctpq] [time]', str(duration[n]), '[off] 0 [pos]', str(tuple(offset_pos))]
            msg.append([time_total, ' '.join(m)])
        return msg

    def load_gestures(self):
        self.all_gestures_files_list = [x for x in os.listdir(self.root_gesture_dir) if 'pos' in x]
        self.all_gestures_list = []
        for j in self.all_gestures_files_list:
            try:
                curr_xml = ET.parse(join(self.root_gesture_dir, j)).getroot()
                curr_g_name = j.split('.')[0]
                curr_part = curr_xml.attrib['ReferencePart']
                self.check_add(self.gestures_dict, curr_g_name, dict())
                self.check_add(self.gestures_dict[curr_g_name], curr_part, dict())
                curr_num_pos = curr_xml.attrib['TotPositions']
                self.gestures_dict[curr_g_name][curr_part]['durations'] = []
                self.gestures_dict[curr_g_name][curr_part]['positions'] = []
                self.gestures_dict[curr_g_name][curr_part]['velocities'] = []
                self.gestures_dict[curr_g_name]['total_duration'] = 0
                for k in range(int(curr_num_pos)):
                    self.gestures_dict[curr_g_name][curr_part]['durations'].append(
                        float(curr_xml[k].attrib['Timing']))

                    self.gestures_dict[curr_g_name][curr_part]['positions'].append(
                        [float(l.text) for l in curr_xml[k][0].findall('PosValue')])

                    self.gestures_dict[curr_g_name][curr_part]['velocities'].append(
                        [float(l.text) for l in curr_xml[k][1].findall('SpeedValue')])
                tot_duration = sum(self.gestures_dict[curr_g_name][curr_part]['durations'])
                if tot_duration > self.gestures_dict[curr_g_name]['total_duration']:
                    self.gestures_dict[curr_g_name]['total_duration'] = tot_duration
                self.all_gestures_list.append(curr_g_name)
                print "Loaded", j
            except:
                print "Invalid xml", j, "skipped"
        self.all_gestures_list = list(set(self.all_gestures_list))

    @staticmethod
    def close_port(j):
        j.interrupt()
        time.sleep(1)
        j.close()

    def respond(self, command, reply):
        reply.clear()
        action = command.get(0).asString()
        if action == "heartbeat":
            reply.addString('ack')
        # -------------------------------------------------
        elif action == "EXIT":
            reply.addString('ack')
            self.close()
        # -------------------------------------------------
        elif action == "move" or action == "getDuration":
            if command.size() >= 2:
                # get(0) -> move
                # get(1) -> action_name
                action_name = command.get(1).asString()
                if action_name in self.all_gestures_list:
                    print command.toString()
                    ret = self.do_action(action_name, args=command)
                    reply.addString("ack")
                    if action == "getDuration":
                        reply.addDouble(ret)
                else:
                    reply.addString("nack")
                    reply.addString("gesture name " + action_name + "not found.")
                    reply.addString("Valid gestures are: " + ' | '.join(self.all_gestures_list))
            else:
                reply.addString("nack")
                reply.addString("gesture name required.")
                reply.addString("Valid gestures are: " + ' | '.join(self.all_gestures_list))
        else:
            reply.addString("nack")
            reply.addString("Command not recognized")
        return True

    def do_action(self, action_name, args=None):
        msg_list = dict()
        args_list = []
        curr_action = self.gestures_dict[action_name]
        init_duration = None
        mirror_flag = False

        print "Parsing move args"
        if args is not None:
            print "checking size"
            if args.size() > 2:
                print "size greater than 2"
                for l in range(2, args.size()):
                    print args.get(l).asString()
                    args_list.append(args.get(l).asString())
                mirror_flag = "mirror" in args_list
                delay = [g.replace("delay=", "") for g in args_list if "delay=" in g]
                if len(delay) > 0:
                    init_duration = float(delay[0])

        # Convert gesture into ctp commands
        print "Convert to ctp commands"
        for part in [x for x in curr_action.keys() if x in self.parts]:
            part_name = copy.deepcopy(part)
            pos_offset = 0

            print "Mirroring"
            if mirror_flag and "arm" in part:
                if part == "left_arm":
                    part_name = "right_arm"
                else:
                    part_name = "left_arm"

            print "Calculating offset"
            if part == "head":
                pos_offset = self.getCurrPosition("head")
            elif all(v == 0.0 for v in curr_action[part]['positions']):
                pos_offset = self.getCurrPosition(part)
            else:
                pos_offset = None

            print "set duration"
            if init_duration is not None:
                curr_action[part]['durations'][0] = init_duration

            msg_list[part_name] = self.construct_ctp_message(curr_action[part]['durations'],
                                                             curr_action[part]['positions'],
                                                             pos_offset)
            print msg_list[part_name]
        if args.get(0).asString() == "move":
            self.send_ctp_messages(msg_list)
        else:
            return max([m[-1][0] for k, m in msg_list.iteritems()])

    def getCurrPosition(self, part_name):
        currPos = self.monitorPorts[part_name].read()
        tuplePos = []
        for p in range(currPos.size()):
            tuplePos.append(round(currPos.get(p).asDouble(), 2))
        return tuplePos

    def interruptModule(self):
        print "Interrupting"
        self.close()
        return True

    def getPeriod(self):
        return 0.1

    def updateModule(self):
        time.sleep(0.05)
        return True


if __name__ == '__main__':
    yarp.Network.init()
    mod = ICubBodyControl()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
    mod.close()
