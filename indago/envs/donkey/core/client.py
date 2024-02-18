"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SDClient
A base class for interacting with the sdsim simulator as server.
The server will create on vehicle per client connection. The client
will then interact by createing json message to send to the server.
The server will reply with telemetry and other status messages in an
asynchronous manner.
Author: Tawn Kramer
"""
import json
import re
import select
import socket
from threading import Thread

from log import Log


def replace_float_notation(string):
    """
    Replace unity float notation for languages like
    French or German that use comma instead of dot.
    This convert the json sent by Unity to a valid one.
    Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2
    :param string: (str) The incorrect json string
    :return: (str) Valid JSON string
    """
    regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
    regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'

    for regex in [regex_french_notation, regex_end]:
        matches = re.finditer(regex, string, re.MULTILINE)

        for match in matches:
            num = match.group("num").replace(",", ".")
            string = string.replace(match.group("num"), num)
    return string


class SDClient:
    def __init__(self, host, port, socket_local_address, poll_socket_sleep_time=0.05):
        self.msg = None
        self.host = host
        self.port = port
        self.socket_local_address = socket_local_address
        self.poll_socket_sleep_sec = poll_socket_sleep_time
        self.th = None

        self.logger = Log("SDClient")

        # the aborted flag will be set when we have detected a problem with the socket
        # that we can't recover from.
        self.aborted = False
        self.connect()

    def connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connecting to the server
        self.logger.info("Connecting to %s:%d " % (self.host, self.port))
        try:
            self.s.connect((self.host, self.port))
        except ConnectionRefusedError as e:
            raise Exception(
                "Could not connect to server. Is it running? "
                "If you specified 'remote', then you must start it manually."
            )

        # time.sleep(pause_on_create)
        self.do_process_msgs = True
        self.th = Thread(target=self.proc_msg, args=(self.s,), daemon=True)
        self.th.start()

    def send(self, m):
        self.msg = m

    def send_now(self, msg):
        # print("send_now:" + msg)
        self.s.sendall(msg.encode("utf-8"))

    def on_msg_recv(self, j):
        print("got:" + j["msg_type"])

    def stop(self):
        # signal proc_msg loop to stop, then wait for thread to finish
        # close socket
        self.do_process_msgs = False
        if self.th is not None:
            self.th.join()
        if self.s is not None:
            self.s.close()

    def proc_msg(self, sock):
        """
        This is the thread message loop to process messages.
        We will send any message that is queued via the self.msg variable
        when our socket is in a writable state.
        And we will read any messages when it's in a readable state and then
        call self.on_msg_recv with the json object message.
        """
        sock.setblocking(False)
        inputs = [sock]
        outputs = [sock]
        partial = []

        while self.do_process_msgs:
            # without this sleep, I was getting very consistent socket errors
            # on Windows. Perhaps we don't need this sleep on other platforms.
            # time.sleep(self.poll_socket_sleep_sec)

            try:
                # test our socket for readable, writable states.
                readable, writable, exceptional = select.select(inputs, outputs, inputs)

                for s in readable:
                    try:
                        data = s.recv(1024 * 256)
                    except ConnectionAbortedError:
                        print("socket connection aborted")
                        self.do_process_msgs = False
                        break

                    # we don't technically need to convert from bytes to string
                    # for json.loads, but we do need a string in order to do
                    # the split by \n newline char. This seperates each json msg.
                    data = data.decode("utf-8")
                    msgs = data.split("\n")

                    for m in msgs:
                        if len(m) < 2:
                            continue
                        last_char = m[-1]
                        first_char = m[0]
                        # check first and last char for a valid json terminator
                        # if not, then add to our partial packets list and see
                        # if we get the rest of the packet on our next go around.
                        if first_char == "{" and last_char == "}":
                            # Replace comma with dots for floats
                            # useful when using unity in a language different from English
                            m = replace_float_notation(m)
                            try:
                                j = json.loads(m)
                                self.on_msg_recv(j)
                            except Exception as e:
                                print("Exception:" + str(e))
                                print("json: " + m)
                        else:
                            partial.append(m)
                            # logger.info("partial packet:" + m)
                            if last_char == "}":
                                if partial[0][0] == "{":
                                    assembled_packet = "".join(partial)
                                    assembled_packet = replace_float_notation(
                                        assembled_packet
                                    )
                                    second_open = assembled_packet.find('{"msg', 1)
                                    if second_open != -1:
                                        # hmm what to do? We have a partial packet. Trimming just
                                        # the good part and discarding the rest.
                                        print(
                                            "got partial packet:"
                                            + assembled_packet[:20]
                                        )
                                        assembled_packet = assembled_packet[
                                            second_open:
                                        ]

                                    try:
                                        j = json.loads(assembled_packet)
                                        self.on_msg_recv(j)
                                    except Exception as e:
                                        print("Exception:" + str(e))
                                        print("partial json: " + assembled_packet)
                                else:
                                    print("failed packet.")
                                partial.clear()

                for s in writable:
                    if self.msg is not None:
                        s.sendall(self.msg.encode("utf-8"))
                        self.msg = None
                if len(exceptional) > 0:
                    print("problems w sockets!")

            except Exception as e:
                print("Exception:", e)
                self.aborted = True
                self.on_msg_recv({"msg_type": "aborted"})
                break
