from PIL import Image

with Image.open('neuralbody.gif') as im:
  for i in range(230):
    im.seek(i)
    im.save('gif/{}.png'.format(i))