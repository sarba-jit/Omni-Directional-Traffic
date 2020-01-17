#!/bin/bash
#!/home/labgforce/.virtualenvs/ai/bin/python


# for entry in video/*
# do
# 	python OmniVidBeeCount.py "$(basename "$entry")"
# done

for entry in applied_science_video/*
do
	python OmniVidBeeCount.py "$(basename "$entry")"
done