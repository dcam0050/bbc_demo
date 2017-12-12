#!/usr/bin/env python

import sys
import time
import readline
import warnings
import numpy as np
import yarp
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for
import os
import threading
import snowboydecoder
import speech_recognition as sr
warnings.simplefilter("ignore")
np.set_printoptions(precision=2)


class sentence_tokenizer(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)
        self.interrupted = False
        self.portsList = dict()
        self.hotword_detector = None
        self.hotword_model = None
        self.token_out_port = None
        self.audio_source = None
        self.tok_validator = None
        self.tokenizer = None
        self.player = None
        self.pause_tokenizer = False

        self.echo_enabled = False
        self.trigger_echo = False
        self.echo_thread = None
        self.hotword_enabled = True

        # Hotword settings
        self.hotword_sensitivity = 0.5
        self.hotword_loop_time = 0.03
        self.hotword_model = os.environ['HOTWORD_MODEL']

        # Tokenizer Settings
        # self.tok_record_duration = None means indefinite
        self.tok_record_duration = None
        self.tok_energy_threshold = 40 #60
        self.tok_window = 0.01 # 0.01
        self.tok_window_rate = 1. / self.tok_window
        self.tok_min_len = 0.5 * self.tok_window_rate
        self.tok_max_len = int(5 * self.tok_window_rate)
        self.tok_max_silence_duration = 0.7 * self.tok_window_rate
        self.tokenizer_mode = None
        self.bdata = None
        self.google_credentials = None

        # Google ASR
        self.use_google = True
        self.asr = None
        self.time_total = 0
        self.num_recs = 0
        self.phrases = ["Hello i cub",
                        "Goodbye i cub",
                        "i cub",
                        "Tony",
                        "Daniel"]

    def configure(self, rf):
        # Setting up rpc port
        self.portsList["rpc"] = yarp.Port()
        self.portsList["rpc"].open("/sentence_tokenizer/rpc:i")
        self.attach(self.portsList["rpc"])

        self.portsList["audio_out"] = yarp.BufferedPortBottle()
        self.portsList["audio_out"].open("/sentence_tokenizer/audio:o")

        # Setting up hotword detection
        self.hotword_detector = snowboydecoder.HotwordDetector(self.hotword_model, sensitivity=self.hotword_sensitivity)

        # Setting up audio tokenizer to split sentences
        self.audio_source = ADSFactory.ads(record=True, max_time=self.tok_record_duration, block_dur=self.tok_window)
        self.tok_validator = AudioEnergyValidator(sample_width=self.audio_source.get_sample_width(),
                                                  energy_threshold=self.tok_energy_threshold)
        self.tokenizer_mode = StreamTokenizer.DROP_TRAILING_SILENCE
        self.tokenizer = StreamTokenizer(validator=self.tok_validator,
                                         min_length=self.tok_min_len,
                                         max_length=self.tok_max_len,
                                         max_continuous_silence=self.tok_max_silence_duration,
                                         mode=self.tokenizer_mode)

        if self.echo_enabled:
            self.echo_thread = threading.Thread(target=self.replayAudio)
            self.echo_thread.start()

        if self.hotword_enabled:
            print("Waiting for hotword to start interaction")
            # self.hotword_detector.start(detected_callback=self.detected_callback,
            #                             interrupt_check=self.interrupt_callback,
            #                             sleep_time=self.hotword_loop_time)
            print("Hotword detected. Starting tokenizer thread")
        else:
            print "Starting tokenizer thread"

        self.asr = sr.Recognizer()

        with open('google_credentials.json', 'r') as credentials:
            self.google_credentials = credentials.read()
        return True

    def detected_callback(self):
        print("Hotword 'Hello iCub' detected")
        self.interrupted = True

    def tok_callback(self, data, start, end, starting=False):
        if data is None:

            audio_bottle = self.portsList["audio_out"].prepare()
            audio_bottle.clear()
            audio_bottle.addString("speaking")
            if starting:
                print "Speaking start"
                audio_bottle.addString("start")
            else:
                print "Speaking stop"
                audio_bottle.addString("stop")
            self.portsList["audio_out"].write()
        else:
            print("Acoustic activity at: {0}--{1}".format(start, end))
            # print "Chunk segmented", time.time()
            # print "Pause value is: ", self.pause_tokenizer
            if not self.pause_tokenizer:
                self.bdata = b''.join(data)

                if self.use_google:
                    audio = sr.AudioData(self.bdata, self.audio_source.get_sampling_rate(),
                                         self.audio_source.get_sample_width())
                    t3 = time.time()
                    try:

                        sentence = self.asr.recognize_google_cloud(audio_data=audio,
                                                                   credentials_json=self.google_credentials,
                                                                   language="en-UK",
                                                                   preferred_phrases=self.phrases)

                        t4 = time.time()
                        dur = t4 - t3
                        self.time_total += dur
                        self.num_recs += 1
                        print sentence, " | Time taken=", dur, " | Mean Time=", self.time_total/self.num_recs
                        audio_bottle = self.portsList["audio_out"].prepare()
                        audio_bottle.clear()
                        audio_bottle.addString("spoken")
                        audio_bottle.addString(str(sentence))
                        self.portsList["audio_out"].write()
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))
                else:
                    audio_bottle = self.portsList["audio_out"].prepare()
                    audio_bottle.clear()
                    audio_bottle.addString("classify")
                    audio_bottle.addString(self.bdata)
                    audio_bottle.addInt(self.audio_source.get_sampling_rate())
                    self.portsList["audio_out"].write()

                if self.echo_enabled:
                    self.trigger_echo = True

    def tokenizerThread(self):
        self.audio_source.open()
        self.tokenizer.tokenize(self.audio_source, callback=self.tok_callback)

    def replayAudio(self):
        self.player = player_for(self.audio_source)
        while True:
            if self.trigger_echo:
                self.player.play(self.bdata)
                self.trigger_echo = False
            time.sleep(2)

    def close(self):
        print('Exiting ...')
        time.sleep(2)
        self.hotword_detector.terminate()
        self.audio_source.close()

        if self.echo_enabled:
            self.player.stop()

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
        elif action == "pause":
            self.pause_tokenizer = True
            print "pausing tokenizer sending"
            reply.addString('ack')
        elif action == "resume":
            self.pause_tokenizer = False
            print "resuming tokenizer sending"
            reply.addString('ack')
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
        self.tokenizerThread()
        print "starting again"
        time.sleep(0.05)
        return True


if __name__ == '__main__':
    yarp.Network.init()
    mod = sentence_tokenizer()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
