#!/bin/bash

export fn=$(echo $1|sed 's/_Analysis\.py/_out.txt/')
echo python3 $1 > ${fn}
