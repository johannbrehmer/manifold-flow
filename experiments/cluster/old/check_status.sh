#!/bin/bash

echo ""
echo "Checking status of all log files"
echo ""

for filename in log_*.log; do
    if tail -n 1 $filename | grep -q "Have a nice day"
    then
        echo "   :)   - ${filename}"
    else
        if grep -Fxq "Error" $filename
        then
            echo "ERROR   - ${filename}"
        else
            echo "running - ${filename}"
        fi
    fi
done

echo ""
