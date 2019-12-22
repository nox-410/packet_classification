#!/bin/bash

# param 1 rule_number
# param 2 trace_number

rule_file_name=./data/rule_$1.rule
trace_prefix=./data/rule

if [ $# -ne 2 ]
then
	echo Usage: $0 \<rule_number\> \<trace_number\>
	exit 1
fi

if [ ! -d "./data" ]
then
	mkdir data
fi
rm -rf data/*

echo Generating rule...
./bin/db_generator -bc ./parameter_files/acl1_seed $1 2 -0.5 0.1 $rule_file_name


threshold=$((100000/$1 + 1))

echo Generating Trace...
for i in $(seq 1 $2)
do
	./bin/trace_generator 1 0 $threshold $rule_file_name
	mv ${rule_file_name}_trace ${trace_prefix}_${1}_${i}.trace
done

rm gmon.out
