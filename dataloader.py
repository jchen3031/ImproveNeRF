import glob


number_classes = len(glob.glob("imageclassifer/train/*"))

print(number_classes)
