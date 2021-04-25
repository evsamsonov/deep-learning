#!/bin/bash
rustc "$1"
# todo проверить ошибку

exec_file=$(echo "$1" | cut -f 1 -d '.')
./"$exec_file"
rm "$exec_file"