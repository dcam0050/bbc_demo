#!/usr/bin/env python

import sys
import time
import readline
import warnings
import numpy as np
import sox
import thread
import yarp
from deepspeech.model import Model
import os
from os.path import join
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class deepSpeechToText(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)
        self.iCub = None
        self.interrupted = False
        self.portsList = dict()
        self.ds_root_dir = None
        self.ds_model_file = None
        self.ds_alphabet_file = None
        self.ds_language_model = None
        self.ds_trie = None
        self.ds_model = None
        self.tokenizer_ctrl_bottle = None
        self.t = [0]*3
        self.ds_root_dir = os.environ['DEEPSPEECH_DIR']

        # self.sentence_list = []

        # Deep Speech settings
        self.ds_BEAM_WIDTH = 500 # 500
        self.ds_LM_WEIGHT = 1.75 # 1.75
        # the following weights are relative to each other not absolute
        self.ds_WORD_COUNT_WEIGHT = 1.00 # 1.00
        self.ds_VALID_WORD_COUNT_WEIGHT = 1.00 # 1.00
        self.ds_N_FEATURES = 26
        self.ds_N_CONTEXT = 9
        self.busy = False
        self.currPos = 0
        self.list_of_classifications = []
        self.my_mutex = thread.allocate_lock()

    def configure(self, rf):
        # Setting up rpc port

        self.portsList["rpc"] = yarp.Port()
        self.portsList["rpc"].open("/deepSpeechToText:rpc:i")
        self.attach(self.portsList["rpc"])
        yarp.Network.connect("/sentence_tokenizer/audio:o", self.portsList["rpc"].getName())

        self.portsList["text_out"] = yarp.BufferedPortBottle()
        self.portsList["text_out"].open("/deepSpeechToText/text:o")

        self.portsList["tokenizer_rpc"] = yarp.BufferedPortBottle()
        self.portsList["tokenizer_rpc"].open("/deepSpeechToText/tokenizer/rpc:o")
        self.tokenizer_ctrl_bottle = self.portsList["tokenizer_rpc"].prepare()
        yarp.Network.connect(self.portsList["tokenizer_rpc"].getName(), "/sentence_tokenizer/rpc:i")

        # Setting up deep speech model client
        self.ds_model_file = join(self.ds_root_dir, "output_graph.pb")
        self.ds_alphabet_file = join(self.ds_root_dir, "alphabet.txt")
        self.ds_language_model = join(self.ds_root_dir, "lm.binary")
        self.ds_trie = join(self.ds_root_dir, "trie")

        self.ds_model = Model(self.ds_model_file, self.ds_N_FEATURES, self.ds_N_CONTEXT,
                              self.ds_alphabet_file, self.ds_BEAM_WIDTH)

        self.ds_model.enableDecoderWithLM(self.ds_alphabet_file,
                                          self.ds_language_model,
                                          self.ds_trie,
                                          self.ds_LM_WEIGHT,
                                          self.ds_WORD_COUNT_WEIGHT,
                                          self.ds_VALID_WORD_COUNT_WEIGHT)
        print "Model loaded and ready to start"
        return True

    def close(self):
        print('Exiting ...')
        time.sleep(2)

        for j in self.portsList.keys():
            self.close_port(self.portsList[j])

        return True

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
        elif action == "classify":
            if command.size() == 3:
                # get(0) -> classify
                # get(1) -> data string
                # get(2) -> sampling rate
                try:
                    self.t[0] = time.time()
                    self.list_of_classifications.append([command.get(1).asString(),
                                                         command.get(2).asInt()])
                except Exception as e:
                    print e
                reply.addString('ack')
            else:
                reply.addString('nack')
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

    def updateModule(self):
        if self.currPos < len(self.list_of_classifications):
            curr_recording = self.list_of_classifications[self.currPos]
            try:
                if len(curr_recording[0]) % 2 != 0:
                    curr_recording[0] += '\x00'
                self.t[1] = time.time()
                converted_data = np.fromstring(curr_recording[0], np.int16)
                this_sentence = self.ds_model.stt(converted_data, curr_recording[1])
                self.t[2] = time.time()
                print "Time taken = ", self.t[2]-self.t[0]
                print "Returned sentence: " + this_sentence
                if len(this_sentence) != 0:
                    sentence_bottle = self.portsList["text_out"].prepare()
                    sentence_bottle.clear()
                    sentence_bottle.addString("spoken")
                    sentence_bottle.addString(this_sentence)
                    self.portsList["text_out"].write()
            except Exception as e:
                print e
                pass
            self.currPos += 1
        else:
            # if len(self.list_of_classifications) != 0:
            #     self.my_mutex.acquire()
            #     print "update module mutex acquired"
            #     self.currPos = 0
            #     self.list_of_classifications = []
            #     self.my_mutex.release()
            #     print "update module mutex released"
            time.sleep(0.05)
        return True


if __name__ == '__main__':
    yarp.Network.init()
    mod = deepSpeechToText()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
