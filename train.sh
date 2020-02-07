#!/bin/bash

function help(){
    printf "Usage :\n";
    printf "    $0 <model_name> <arguments>\n"
    exit;
}

if [[ $# < 1 ]]
then
    help;
fi

file="train_$1";

if [ ! -f "training/$file.py" ]
then
    printf "File ${file} not found\n";
    printf "enter correct model name \n";
    exit;
fi

python -m training.$file "${@:2}";
