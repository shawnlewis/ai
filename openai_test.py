#!/bin/python

import cv2
import gym

#env = gym.make('SpaceInvaders-v0')

print gym.envs.registry.all()

for envspec in gym.envs.registry.all():
  print envspec
  env = envspec.make()
  env.reset()
  while 1:
    try:
      env.render()
    except gym.error.UnsupportedMode:
      break
    image = env.render(mode='rgb_array')
    image = cv2.resize(image, (0, 0), fx=4.0, fy=4.0)
    image = image[:, :, ::-1]
    cv2.imshow('main', image)
    cv2.waitKey(1)
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
      break
