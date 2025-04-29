#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_video_file>"
  exit 1
fi

input="$1"
output_file="$(basename "$input")"
ffmpeg -i "$input" -vf "scale=1920:1080" -c:v libx264 -preset ultrafast -crf 30 "$output_file"