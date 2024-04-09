# Copyright (C) 2023 Intel Corporation


# SPDX-License-Identifier: Apache-2.0


import logging
from logging.handlers import RotatingFileHandler


class ConvAssistLogger:
    """
        Create an instance to log messages to a file

    """

    def __init__(self, filename, filepath, level):
        self.file_name = filename
        self.file_path = filepath
        self.app_log = None
        self.isLogInitialized = False
        self.level = level
        self.my_handler = None

    def setLogger(self):
        """
            Set the Main object of the logger

        """
        try:
            log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            if len(self.file_path) > 1:
                logFile = self.file_path + "\\" + self.file_name + ".txt"
                self.my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=5 * 1024 * 1024,
                                                 backupCount=2, encoding=None, delay=0)
                self.my_handler.setFormatter(log_formatter)
                self.my_handler.setLevel(self.level)
                self.app_log = logging.getLogger(self.file_name)
                self.app_log.setLevel(self.level)
                self.app_log.addHandler(self.my_handler)
                self.isLogInitialized = True
            else:
                self.isLogInitialized = False
        except Exception as e:
            self.isLogInitialized = False

    def Log(self, message):
        """ 
            Add the Message to the log file

        Args:
            message (string): Message to log
        """
        if self.isLogInitialized:
            self.app_log.info(message)

    def IsLogInitialized(self):
        """
            gets if the log object is initialized

        Returns:
            bool: Is the logger initialized
        """
        return self.isLogInitialized

    def Close(self):
        """
            Clears and close the handler used to log into the file
        """
        self.app_log.removeHandler(self.my_handler)
        self.my_handler.close()
