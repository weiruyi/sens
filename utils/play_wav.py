import pygame
pygame.init()

pygame.mixer.music.load('../underWater/data_1/origin_wav/left01.wav')
pygame.mixer.music.play()
pygame.event.wait()