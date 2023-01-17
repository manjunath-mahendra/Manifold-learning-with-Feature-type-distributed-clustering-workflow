#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time


def count(testFn, items):
    s = 0
    for x in items:
        if testFn(x):
            s += 1
    return s



class Timing:
    def __init__(self, name="Duration"):
        self.name = name
        self.tStart = time.process_time()
        self.tStepStart = self.tStart

    def step(self, message=""):
        now = time.process_time()
        duration = now - self.tStart
        durationStep = now - self.tStepStart
        self.tStepStart = now

        if message == "":
            print(f"{self.name}: {durationStep:0.5f} / {duration:0.3f}s")
        else:
            print(f"{self.name} ({message}): {durationStep:0.5f} / {duration:0.3f}s")
        return duration


# In[ ]:




