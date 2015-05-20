import argparse, subprocess, os.path


def extract(args):
	with open(args.features, 'w') as f_file:
		f_line = f.readline()
		classes = f_line.split(';')[1:]
		for f_line in f_file:
			line_args = f_line.split(';')
			file_name = basename(line_args[0)
			values = line_args[1:]
			make_args(classes, values)
			call_string = "SMILExtract -C {} -I {} -0 {} -instname {} -corpus {} -arfftargetsfile {} "
				.format(config, file_name, args.output, args.corpus, arfftargets)
			ret = subprocess.call(call_string + make_args(classes, values))
			if(ret):
				print "."
			else:
				return 1
	return 0

def make_args(classes, values):
	return " ".join([make_flag(class, value) for class, value in zip(classes, values[1:])])

def make_flag(class, value):
	return "-" + class + " " + value


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('corpus', type=dir, help="Path to the corpus")
	parser.add_argument('config', type=file, help="Path to the config file in format required by openSMILE")
	parser.add_argument('-o', '--output', type=argparse.FileType('w'), default="output.arff", help="Path to the output arff file")
	parser.add_argument('--arfftargets', type=file, default="arff_targets_gui_labels.conf.inc", help="Path to the config file with the labels")
	parser.add_argument('-f', '--features', type=file, help="Path to the csv file containing classes")
	ret = extract(parser.parse_args())
	exit(ret)

