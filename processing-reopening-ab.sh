#!/bin/sh

make
./reopening # single-core, takes ~36 hours @ Ryzen 9 5950X
python3 compute_all_10_18.py # multi-core
python3 compute_all_18_end.py # multi-core, requires ~5TB of disk space
python3 postprocessing1_aggregate_by_disccount_and_dedup.py # multi-core, requires ~5TB of disk space
python3 postprocessing2_concat_and_sort.py #multi-core, requires ~3TB of disk space
python3 postprocessing3.py # single-core, requires 500GB of disk space
python3 postprocessing4.py # single-core, requires 500GB of disk space
