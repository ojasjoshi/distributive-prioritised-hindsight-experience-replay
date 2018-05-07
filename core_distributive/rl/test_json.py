import json

def process_json(file_path):
	data = []
	last_json_object = ""
	with open(file_path) as f:
	    for line in f:
	    	for character in line:
	    		if(character=='{'):
	    			temp_object = ""
	    		if(character=='}'):
	    			temp_object += character
	    			last_json_object=temp_object
	    		temp_object += character
	new_json_file = open("data/temp.json", "w")
	new_json_file.write(last_json_object)
	new_json_file.close()
	data = json.load(open("data/temp.json"))
	return data
