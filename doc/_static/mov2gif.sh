#!/bin/sh
ffmpeg -i $1 -pix_fmt rgb8 -r 10 output.gif
