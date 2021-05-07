#!/bin/bash
if ! rustc --edition 2018 "$1"; then
    echo "Failed to compile"
    return
fi

exec_file=$(echo "$1" | cut -f 1 -d '.')
./"$exec_file"
rm "$exec_file"