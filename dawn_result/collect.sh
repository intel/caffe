#!/bin/bash

cores=$(lscpu | grep "Core(s) per socket" |awk -F ' ' '{ print $4 }')
for log in $(ls |grep "[0-9].log" |sort -t '_' -k 5 -n)
do
    start_time=$(date -d "$(head -1 $log |awk '{print $2}')" +%s)
    end_time=$(date -d "$(tail -1 $log |awk '{print $2}')" +%s)
    if [ $start_time -gt $end_time ];then
        end_time=$[${end_time}+24*3600]
    fi
    total_time=$[${end_time}-${start_time}]

    top1=$(grep "loss3/top-1" $log |tail -1 |sed "s/.*= //")
    top5=$(grep "loss3/top-5" $log |tail -1 |sed "s/.*= //")

    echo -e "
        cores: $cores\t time: ${total_time}s\t top1: ${top1}\t top5: ${top5}"
done

