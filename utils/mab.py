#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

class UCBMultiArmBandit:
    def __init__(self, num_arms, init_rewards):
        self.num_arms = num_arms
        self.total_rewards = np.zeros(num_arms)
        for i in range(len(init_rewards)):
            self.total_rewards[i] = init_rewards[i]
        self.arm_counts = np.zeros(num_arms)
        self.total_plays = 0

    def select_arm(self):
        if 0 in self.arm_counts:
            # Play each arm once to initialize
            return np.argmin(self.arm_counts)
        else:
            ucb_values = self.total_rewards / self.arm_counts + np.sqrt(2 * np.log(self.total_plays) / self.arm_counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_rewards[chosen_arm] += reward
        self.arm_counts[chosen_arm] += 1
        self.total_plays += 1