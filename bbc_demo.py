#!/usr/bin/env python

import sys
import time
import readline
import warnings
import numpy as np
import yarp
import icubclient
import simplejson
import requests
import random
from enum import Enum
from os.path import join
import os
from pydub import AudioSegment
from pydub.playback import play
import re
from io import BytesIO
import urllib

try:
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Metadata, Interpreter
    use_rasa_grammar = True
except ImportError:
    use_rasa_grammar = False

warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class bbc_demo(yarp.RFModule):
    def __init__(self, userasa):
        yarp.RFModule.__init__(self)
        self.iCub = None
        self.interrupted = False
        self.portsList = dict()
        self.rpcPort = None
        self.received_text = None
        self.tokenizer_state = True
        self.list_of_actions = []
        self.list_of_emotions = []
        self.num_actions = None
        self.num_emotions = None
        self.prepared_action_dict = dict()
        self.duration_action_dict = dict()
        self.attention_on_agent = False
        self.t_last_speak = time.time()
        self.tspeak_timer = 10
        self.agent_speaking = True

        self.homeoPortName = "/manager/toHomeostasis/rpc" + ":o"
        self.homeoRPC = "/homeostasis/rpc"

        # iCubClient parameters
        self.speech_client = None
        self.emotion_client = None
        self.sam_client = None
        self.withProactive = False
        self.use_tacotron = False
        self.planned_reply = ''
        self.processed_text = True
        # Rasa Parameters
        if userasa:
            self.rasa_root_dir = "/home/icub/user_files/bbc_demo/rasa_development/rasa_files"
            self.rasa_data_file = "demo-rasa.json"
            self.rasa_model_dir = "models/default"
            self.rasa_config_file = "config_spacy.json"
            self.rasa_config = None
            self.rasa_interpreter = None
            self.use_rasa_grammar = True
        else:
            self.use_rasa_grammar = False

        # Chat interface settings
        version = 3
        self.chat_interface = "TalkML"
        self.grammar_dict = dict()
        self.detected_grammar = {"g1": '', "g2": ''}
        self.grammars = dict()
        self.grammars_mult = dict()
        self.TKML_conf_dir = "/home/icub/user_files/bbc_demo/talkml_bbc"
        self.TalkML_tkml_file = join(self.TKML_conf_dir, "TonyInterview_v" + str(version) + ".tkml")
        self.TalkML_grammar_file = join(self.TKML_conf_dir, "TonyInterview_v" + str(version) + ".gmr")
        self.DID_file = join(self.TKML_conf_dir, "tkml_did.txt")
        with open(self.DID_file, 'r') as myfile:
            self.TalkML_D_id = myfile.read().replace('\n', '')

        self.TalkML_S_id = ''
        self.TalkML_URL = "http://www.proseco.co.uk/prosebot"
        self.TalkML_headers = {'Content-type': 'application/json',
                               'DId': 'proseco.' + self.TalkML_D_id,
                               'SId': 'proseco.' + self.TalkML_S_id}
        self.TalkML_timeout = dict()
        self.TalkML_timeout['no_input'] = 7
        self.TalkML_timeout['no_match'] = 4
        self.TalkML_timeout['heard'] = 1.5
        self.TalkML_currState = None
        self.TKML_waitForInput = False
        self.TKML_stop = False
        self.delay_sleep = 0.1
        self.tokenizer_delay = 0.3

        class TKML_States(Enum):
            WAIT2TALK = 0
            TALKING = 1
            WAIT2HEAR = 2
            HEARING = 3
            HEARD = 4

        class TKML_Actions(Enum):
            upload = 'upload',
            start = 'start',
            heard = 'heard',
            noinput = 'noinput',
            nomatch = 'nomatch',
            getSayNext = 'getSayNext'

        self.TKML_Actions = TKML_Actions

        self.TKML_States = TKML_States

        self.TalkML_enabled = False
        self.grammar_mode = "not rasa"

    def configure(self, rf):
        # # --------------------------------Testing---------------------------------
        # self.list_of_actions = ["left_arm_gun_pt1", "fast_long_wave", "left_arm_outstretched",
        #                         "hand_up_5_fingers_palm_outward", "left_arm_kill", "left_arm_outstretched_you_got_it",
        #                         "hand_up_5_fingers_palm_inward", "left_arm_outstretched_thumbs_up", "left_arm_kill"]
        # self.num_actions = len(self.list_of_actions)
        #
        # self.list_of_emotions = ["neutral", "talking", "happy", "sad", "surprised", "evil", "angry", "shy", "cunning"]
        # self.num_emotions = len(self.list_of_emotions)
        #
        # for n in self.list_of_actions:
        #     self.prepared_action_dict[n] = self.prepare_movement(n)
        #
        # r = self.get_chatbot_reply("This is a sentence")
        # self.parse_chatbot_reply(r)
        # # ------------------------------------------------------------------------

        self.withProactive = rf.find('withProactive').toString_c().lower() == "true"

        # Open BBC rpc port
        self.portsList["rpc"] = yarp.Port()
        self.portsList["rpc"].open("/BBC_Demo/rpc:i")
        self.attach(self.portsList["rpc"])
        yarp.Network.connect("/sentence_tokenizer/audio:o", self.portsList["rpc"].getName())
        # yarp.Network.connect("/deepSpeechToText/text:o", self.portsList["rpc"].getName())

        # Open Body control port
        self.portsList["body_control"] = yarp.RpcClient()
        self.portsList["body_control"].open("/BBC_Demo/body_control/rpc:o")
        yarp.Network.connect(self.portsList["body_control"].getName(), "/body_control/rpc:i")

        # Open speech dev control port
        self.portsList["speech_control"] = yarp.RpcClient()
        self.portsList["speech_control"].open("/BBC_Demo/speech_control/rpc:o")
        yarp.Network.connect(self.portsList["speech_control"].getName(), "/icub/speech:rpc")

        # Open tokenizer control port
        self.portsList["tokenizer_control"] = yarp.RpcClient()
        self.portsList["tokenizer_control"].open("/BBC_Demo/sentence_tokenizer/rpc:o")
        yarp.Network.connect(self.portsList["tokenizer_control"].getName(), "/sentence_tokenizer/rpc:i")

        self.list_of_actions = ["left_arm_gun_pt1", "fast_long_wave", "left_arm_outstretched",
                                "hand_up_5_fingers_palm_outward", "left_arm_kill", "left_arm_outstretched_you_got_it",
                                "hand_up_5_fingers_palm_inward", "left_arm_outstretched_thumbs_up", "left_arm_kill"]
        self.num_actions = len(self.list_of_actions)

        self.list_of_emotions = ["neutral", "talking", "happy", "sad", "surprised", "evil", "angry", "shy", "cunning"]
        self.num_emotions = len(self.list_of_emotions)

        rep = yarp.Bottle()
        for n in self.list_of_actions:
            rep.clear()
            self.prepared_action_dict[n] = self.prepare_movement(n)
            cmd = self.prepare_movement(n, duration=True)
            self.portsList["body_control"].write(cmd, rep)
            self.duration_action_dict[n] = rep.get(1).asDouble()

        if self.withProactive:
            self.portsList["toHomeo"] = yarp.Port()
            self.portsList["toHomeo"].open(self.homeoPortName)
            yarp.Network.connect(self.homeoPortName, self.homeoRPC)

        # Setup icub client
        self.iCub = icubclient.ICubClient("BBC_Demo", "icubClient", "BBC_demo.ini")
        if not self.iCub.connect():
            print "iCub not connected"
            # return False

        if not self.iCub.getSpeechClient():
            print "Speech not connected"
            # return False
        else:
            self.speech_client = self.iCub.getSpeechClient()
            self.speech_client.Connect()

        if not self.iCub.getEmotionClient():
            print "Emotion not connected"
            # return False
        else:
            self.emotion_client = self.iCub.getEmotionClient()
            self.emotion_client.Connect()

        # if not self.iCub.getSAMClient():
        #     print "SAM not connected"
            # return False
        # else:
        #     self.sam_client = self.iCub.getSAMClient()
        #     self.sam_client.Connect()
        # Setting up speech if available

        # Setting up Rasa Grammar
        if self.grammar_mode == "rasa":
            search_dir = join(self.rasa_root_dir, self.rasa_model_dir)
            model_dirs = [join(search_dir, d) for d in os.listdir(search_dir) if os.path.isdir(join(search_dir, d))]
            self.rasa_model_dir = max(model_dirs, key=os.path.getmtime)

            self.rasa_config = RasaNLUConfig(join(self.rasa_root_dir, self.rasa_config_file))
            self.rasa_interpreter = Interpreter.load(join(self.rasa_root_dir, self.rasa_model_dir),
                                                     self.rasa_config)
        else:
            with open(self.TalkML_grammar_file) as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for j in content:
                parts = j.split('.*')
                key = parts[0].replace('\t', '')
                vals = parts[1][1:-1].split('|')
                for v in vals:
                    if "*" in v:
                        try:
                            self.grammars_mult[key].append(v.split("*"))
                        except:
                            self.grammars_mult[key] = [v.split("*")]
                    else:
                        try:
                            self.grammars[key].append(v)
                        except:
                            self.grammars[key] = [v]

            # Setting up chat interface
        if self.chat_interface == "TalkML":
            # Reading in tkml file
            with open(self.TalkML_tkml_file, "r") as tkml_file:
                self.tkml_def = tkml_file.read().replace("\n", " ")

            # Setting up upload request body
            startup_request = \
                {"version": "1.0",
                 "action": self.TKML_Actions.upload.value,
                 "tkml": self.tkml_def}

            # Sending startup request
            startup_reply = self.TalkML_Send(startup_request)
            if str(startup_reply) == "<Response [200]>":
                print "Upload successful"
                # Sending start
                start_request = \
                    {
                        "action": "start",
                        "version": "1.0"
                    }
                start_reply = self.TalkML_Send(start_request)
                self.TalkML_enabled = self.parse_chatbot_reply(start_reply)
                if self.TalkML_enabled:
                    print "Start successful"
                else:
                    print "TalkML reply: ", start_reply
                    print "Start unsuccessful"
            else:
                print "TalkML reply: ", startup_reply
                if startup_reply is not None:
                    print startup_reply._content
                print "TalkML upload unsuccessful"
                return False
        elif self.chat_interface == "testing":
            r, _ = self.get_chatbot_reply("Hello")
            self.parse_chatbot_reply(r)

        return True

    def check_grammar(self, grm, text):
        text = text.lower()
        b2 = False
        b1 = False

        if grm in self.grammars.keys():
            b1 = any(wrd in text for wrd in self.grammars[grm])

        if grm in self.grammars_mult.keys():
            for k in self.grammars_mult[grm]:
                b2 = b2 or all(wrd in text for wrd in k)

        if b1 or b2:
            return True
        else:
            return False

    def TalkML_Send(self, message):
        print message
        return requests.post(self.TalkML_URL, data=simplejson.dumps(message), headers=self.TalkML_headers)

    def freeze_drives(self):
        # Prepare command
        cmd = yarp.Bottle()
        cmd.clear()
        cmd.addString('freeze')
        cmd.addString('all')
        # Send command
        if not yarp.Network.isConnected(self.homeoPortName, self.homeoRPC):
            print yarp.Network.connect(self.homeoPortName, self.homeoRPC)
            time.sleep(0.1)
        self.toHomeo.write(cmd)

    def unfreeze_drives(self):
        # Prepare command
        cmd = yarp.Bottle()
        cmd.clear()
        cmd.addString('unfreeze')
        cmd.addString('all')
        # Send command
        if not yarp.Network.isConnected(self.homeoPortName, self.homeoRPC):
            print yarp.Network.connect(self.homeoPortName, self.homeoRPC)
            time.sleep(0.1)
        self.toHomeo.write(cmd)

    def tacotron_say(self, message):

        if self.use_tacotron:
            print "Saying", message
            ret = requests.get("http://localhost:9000/synthesize?text=" + urllib.quote(message))
            speech_audio = AudioSegment.from_file(BytesIO(ret.content), format="wav")
            play(speech_audio)
        else:
            self.iCub.say(message)

    def get_chatbot_reply(self, sentence, sendToTKML=False):
        if self.chat_interface == "testing":
            parts = sentence.split(" ")

            tog = True
            this_sentence = ""

            # "<say> That's really <gesture>roll eyes</gesture> good.</say>
            this_sentence += "<say> "
            for k in parts:
                this_sentence += k + " "
                if tog:
                    this_sentence += "<gesture>"
                    this_sentence += self.list_of_actions[random.randint(1, self.num_actions)]
                    this_sentence += "</gesture> "
                else:
                    this_sentence += "<emotion>"
                    this_sentence += self.list_of_emotions[random.randint(1, self.num_emotions)]
                    this_sentence += "</emotion> "
                tog = not tog
            this_sentence += "</say>"
            print this_sentence
        elif self.chat_interface == "TalkML":
            # parse grammar
            message_request = None
            this_sentence = None
            currAction = None
            print "current sentence: ", sentence
            if sentence in self.TKML_Actions:
                print "received", sentence
                currAction = sentence.value
            elif type(sentence) is str:
                currAction = self.TKML_Actions.heard.value
                if self.grammar_mode == "rasa":
                    intents = self.rasa_interpreter.parse(unicode(sentence, "utf-8"))
                    if intents['intent']['name'] == '':
                        print "extracted no match"
                        return None, False
                    else:
                        grammar = intents['intent']['name']
                        print "extracted intent:", grammar
                        message_request = \
                        {
                            "action": currAction,
                            "grammar": grammar,
                            "version": "1.0"
                        }
                        # message_request += intents['entities']
                else:
                    print "Grammar parsing: ", sentence
                    message_request = \
                    {
                        "action": currAction,
                        "version": "1.0"
                    }
                    if self.detected_grammar['g1'] == '':
                        if self.grammar_dict['g1'] is not None:
                            for g in self.grammar_dict['g1'].split('|'):
                                if self.detected_grammar['g1'] == '':
                                    if self.check_grammar(g, sentence):
                                        self.detected_grammar['g1'] = g
                                        self.detected_grammar['g2'] = ''
                                    else:
                                        self.detected_grammar['g1'] = ''
                        else:
                            self.detected_grammar['g1'] = ''

                        if self.detected_grammar['g2'] == '':
                            if self.grammar_dict['g2'] is not None and self.detected_grammar['g1'] == '':
                                for g in self.grammar_dict['g2'].split('|'):
                                    if self.detected_grammar['g2'] == '':
                                        if self.check_grammar(g, sentence):
                                            self.detected_grammar['g2'] = g
                                        else:
                                            self.detected_grammar['g2'] = ''
                            else:
                                self.detected_grammar['g2'] = ''
                print "expecting grammar", self.grammar_dict
                print "Detected grammar: ",  self.detected_grammar['g1'], "|",  self.detected_grammar['g2']
                if self.detected_grammar['g1'] != '' and self.planned_reply == '':
                    print "Detected g1. Planning reply"
                    message_request['grammar'] = self.detected_grammar['g1']
                    self.planned_reply = self.TalkML_Send(message_request)
                    print "received reply", self.planned_reply.json()
                elif sendToTKML:
                    print "sending to tkml"
                    if self.detected_grammar['g2'] != '':
                        message_request['grammar'] = self.detected_grammar['g2']
                    else:
                        message_request['action'] = self.TKML_Actions.nomatch.value
                    this_sentence = self.TalkML_Send(message_request)
                    print "received reply", this_sentence.json()
            else:
                print sentence, "invalid input"
                message_request = None

            if message_request is None:
                message_request = \
                {
                    "action": currAction,
                    "version": "1.0"
                }
                this_sentence = self.TalkML_Send(message_request)
        else:
            this_sentence = sentence

        return this_sentence, True

    def parse_chatbot_reply(self, reply):
        if str(reply) == "<Response [200]>":
            reply_json = reply.json()
            self.TalkML_currState = self.TKML_States.WAIT2TALK.value
            sayThis = str(reply_json.get("sayThis"))
            self.grammar_dict['g1'] = reply_json.get("g1")
            self.grammar_dict['g2'] = reply_json.get("g2")

            if self.grammar_dict['g1'] is None:
                self.grammar_dict['g1'] = ''

            if self.grammar_dict['g2'] is None:
                self.grammar_dict['g2'] = ''

            print "Expecting", self.grammar_dict['g1'], "|", self.grammar_dict['g2']

            if len(sayThis) == 0:
                sayThis = None

            self.TKML_waitForInput = False

            if sayThis is None and self.grammar_dict['g1'] == "" and self.grammar_dict['g2'] == "":
                self.TKML_waitForInput = False
                self.TKML_stop = True
            elif sayThis is None and self.grammar_dict['g1'] == "" and self.grammar_dict['g2'] != "":
                self.TKML_waitForInput = True
                self.TKML_stop = False
            elif sayThis is None and self.grammar_dict['g1'] != "" and self.grammar_dict['g2'] == "":
                self.TKML_waitForInput = True
                self.TKML_stop = False
            elif sayThis is None and self.grammar_dict['g1'] != "" and self.grammar_dict['g2'] != "":
                self.TKML_waitForInput = True
                self.TKML_stop = False
            elif sayThis is not None and self.grammar_dict['g1'] == "" and self.grammar_dict['g2'] == "":
                self.TKML_waitForInput = False
                self.TKML_stop = False
            elif sayThis is not None and self.grammar_dict['g1'] == "" and self.grammar_dict['g2'] != "":
                self.TKML_waitForInput = False
                self.TKML_stop = False
            elif sayThis is not None and self.grammar_dict['g1'] != "" and self.grammar_dict['g2'] == "":
                self.TKML_waitForInput = True
                self.TKML_stop = False
            elif sayThis is not None and self.grammar_dict['g1'] != "" and self.grammar_dict['g2'] != "":
                self.TKML_waitForInput = True
                self.TKML_stop = False

            if sayThis is not None and sayThis != "None":
                self.TalkML_currState = self.TKML_States.TALKING.value
                sayThis = sayThis.replace("/", "")
                rep = yarp.Bottle()

                # Turn listening off

                self.toggle_tokenizer()
                time.sleep(self.tokenizer_delay)
                compsent = ""
                for k in sayThis.split(" "):
                    if "<" not in k or ">" not in k:
                        compsent += " " + k
                    elif "gesture" in k:
                        self.tacotron_say(compsent)
                        compsent = ""
                        action = k.replace("<gesture>", "")
                        self.portsList["body_control"].write(self.prepared_action_dict[action], rep)
                        time.sleep(self.duration_action_dict[action])
                    elif "emotion" in k:
                        self.tacotron_say(compsent)
                        compsent = ""
                        em = k.replace("<emotion>", "")
                        self.emotion_client.setEmotion(em, "all")
                if compsent != "":
                    self.tacotron_say(compsent)

                # Turn listening on again
                time.sleep(self.tokenizer_delay)
                self.toggle_tokenizer()
                return True
            else:
                print "Reply is empty"
                return True
        else:
            if reply is not None:
                print reply._content
            print "reply is not 200"
            return False

    def close(self):
        print('Exiting ...')
        time.sleep(2)

        for j in self.portsList.keys():
            self.close_port(self.portsList[j])

        return True

    def toggle_tokenizer(self):
        cmd = yarp.Bottle()
        cmd.clear()
        rep = yarp.Bottle()
        if self.tokenizer_state:
            cmd.addString("pause")
        else:
            cmd.addString("resume")
        self.portsList["tokenizer_control"].write(cmd, rep)
        self.tokenizer_state = not self.tokenizer_state

    @staticmethod
    def prepare_movement(action_name, delay=None, duration=False):
        cmd = yarp.Bottle()
        cmd.clear()
        if duration:
            cmd.addString("getDuration")
        else:
            cmd.addString("move")
        cmd.addString(action_name)
        if delay is not None:
            cmd.addString("delay="+str(delay))
        return cmd

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
        if action == "speaking":
            if command.get(1).asString() == 'start':
                self.agent_speaking = True
                print "------------------------waiting for stop speaking"
            else:
                self.agent_speaking = False
                print "------------------------stopped speaking"
            reply.addString('ack')
        # -------------------------------------------------
        if action == "spoken":
            if command.size() == 2:
                self.agent_speaking = False
                self.received_text = command.get(1).asString()
                print "Received sentence", self.received_text
                self.processed_text = False
                reply.addString('ack')
            else:
                reply.addString("nack")
        # -------------------------------------------------
        elif action == "EXIT":
            reply.addString('ack')
            self.close()
        # -------------------------------------------------
        else:
            reply.addString("nack")
            reply.addString("Command not recognized")
        return True

    def interruptModule(self):
        print "Interrupting"
        self.close()
        return True

    def getPeriod(self):
        return 0.1

    @staticmethod
    def getPauseValue(start_time):
        return time.time() - start_time

    def updateModule(self):

        if not self.TKML_waitForInput:
            start_time = time.time()
            self.detected_grammar = {'g1': '', 'g2': ''}
            self.TalkML_currState = self.TKML_States.WAIT2TALK.value
            while self.agent_speaking:
                if not self.processed_text:
                    self.get_chatbot_reply(self.received_text)
                    self.processed_text = True
                time.sleep(self.delay_sleep)

            if self.detected_grammar != {'g1': '', 'g2': ''}:
                annotated_reply, _ = self.get_chatbot_reply(self.received_text)
            else:
                annotated_reply, _ = self.get_chatbot_reply(self.TKML_Actions.getSayNext)
            self.parse_chatbot_reply(annotated_reply)
        else:
            self.TalkML_currState = self.TKML_States.WAIT2HEAR.value
            start_time = time.time()

            while self.getPauseValue(start_time) < self.TalkML_timeout['no_input'] and \
                  self.TalkML_currState == self.TKML_States.WAIT2HEAR.value:

                time.sleep(self.delay_sleep)

                if self.agent_speaking:
                    self.TalkML_currState = self.TKML_States.HEARING.value

            if self.TalkML_currState == self.TKML_States.WAIT2HEAR.value:

                annotated_reply, _ = self.get_chatbot_reply(self.TKML_Actions.noinput)
                self.parse_chatbot_reply(annotated_reply)

            else:
                start_time = time.time()

                while self.getPauseValue(start_time) < self.TalkML_timeout['no_match'] and \
                      self.TalkML_currState == self.TKML_States.HEARING.value:

                    if not self.processed_text:
                        self.get_chatbot_reply(self.received_text)
                        self.processed_text = True

                    time.sleep(self.delay_sleep)

                    # if self.received_text is None and not self.agent_speaking:
                    #     time.sleep(self.delay_sleep)

                    if self.agent_speaking:
                        start_time = time.time()

                    # print self.getPauseValue(start_time), self.TalkML_timeout['heard']
                    if self.getPauseValue(start_time) > self.TalkML_timeout['heard'] and not\
                       self.agent_speaking and self.detected_grammar['g1'] != '':
                        print "inside"
                        # report g1 grammar
                        # annotated_reply, grammar_match = self.get_chatbot_reply(self.received_text)
                        self.received_text = None
                        self.parse_chatbot_reply(self.planned_reply)
                        self.planned_reply = ''
                        self.detected_grammar = {'g1': '', 'g2': ''}
                        self.TalkML_currState = self.TKML_States.HEARD.value

                if self.TalkML_currState == self.TKML_States.HEARING.value:
                    # report g2 grammar or no match
                    annotated_reply, _ = self.get_chatbot_reply(self.received_text, sendToTKML=True)
                    self.received_text = None
                    self.detected_grammar = {'g1': '', 'g2': ''}
                    self.parse_chatbot_reply(annotated_reply)
        return True


if __name__ == '__main__':
    yarp.Network.init()
    mod = bbc_demo(use_rasa_grammar)
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
