```bash
sudo apt install imagemagick ghostscript
dpkg -l | grep imagemagick
mogrify -format png -trim -resize 1500x1500 -quality 90 *.tif
```
