#!/bin/bash

echo ""
echo "Checking status of all log files"
echo ""

for filename in log_*.log; do
    if grep -Fxq "All done!" $filename
    then
        echo "${filename}: success"
    else
        if grep -Fxq "Exception" $filename
        then
            echo "${filename}: PROBABLY ERROR"
        else
            echo "${filename}: unfinished or ERROR"
        fi
    fi
done

echo ""
