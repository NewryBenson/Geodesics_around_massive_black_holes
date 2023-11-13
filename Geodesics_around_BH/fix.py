import PIL.Image as Image

image = Image.new(mode="RGB", size=(2048, 1024))
pixels = image.load()

for x in range(1024):
    for y in range(512):
        pixels[x, y] = (255, 0, 0)
        pixels[x+1024, y] = (0, 255, 0)
        pixels[x + 1024, y+512] = (0, 0, 255)
        pixels[x, y+512] = (255, 255, 0)


'''for x in range(2048):
    if x % 64 == 0:
        for y in range(1024):
            pixels[x+1, y] = (0, 0, 0)
            pixels[x, y] = (0, 0, 0)
for y in range(1024):
    if y % 64 == 0:
        for x in range(2048):
            pixels[x, y+1] = (0, 0, 0)
            pixels[x, y] = (0, 0, 0)
for y in range(1024):
    pixels[2047, y] = (0, 0, 0)
    pixels[2046, y] = (0, 0, 0)

for x in range(2048):
    pixels[x, 1023] = (0, 0, 0)
    pixels[x, 1022] = (0, 0, 0)'''
image.save("colorCorrect.png")