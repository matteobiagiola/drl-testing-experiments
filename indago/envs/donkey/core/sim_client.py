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
"""

"""
author: Tawn Kramer
date: 9 Dec 2019
file: sim_client.py
notes: wraps a tcp socket client with a handler to talk to the unity donkey simulator
"""
import json

from indago.envs.donkey.core.client import SDClient
from log import Log


class SimClient(SDClient):
    """
    Handles messages from a single TCP client.
    """

    def __init__(self, address, socket_local_address, msg_handler):
        # we expect an IMesgHandler derived handler
        # assert issubclass(msg_handler, IMesgHandler)

        # hold onto the handler
        self.msg_handler = msg_handler

        # connect to sim
        super().__init__(*address, socket_local_address)

        # we connect right away
        msg_handler.on_connect(self)

        self.logger = Log("SimClient")

    def queue_message(self, msg):
        # right now, no queue. Just immediate send.
        json_msg = json.dumps(msg)
        self.send(json_msg)

    def on_msg_recv(self, jsonObj):
        # pass message on to handler
        self.msg_handler.on_recv_message(jsonObj)

    def handle_close(self):
        # when client drops or closes connection
        if self.msg_handler:
            self.msg_handler.on_disconnect()
            self.msg_handler = None
            self.logger.info("Connection dropped")

        self.close()

    def is_connected(self):
        return not self.aborted

    def __del__(self):
        pass
        # self.close()

    def close(self):
        # Called to close client connection
        self.stop()

        if self.msg_handler:
            self.msg_handler.on_close()
