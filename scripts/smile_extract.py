import argparse, subprocess, os.path, csv


def extract(args):
	with open(args.features, 'r') as f_file:
		f_reader = csv.reader(f_file, delimiter=';')
		classes = f_reader.next()[1:]
		for f_line in f_reader:
			instance_name = f_line[0]
			file_name = os.path.join(args.corpus, instance_name + ".wav")
			values = f_line[1:]
			make_args(classes, values)
			call_string = "SMILExtract -C {} -I {} -O {} -instname {} -arfftargetsfile {} ".format(
				args.config, file_name, args.output, instance_name, args.arfftargets)
			print "."
			try:
				print call_string + make_args(classes, values)
				subprocess.check_output(call_string + make_args(classes, values), shell=True)
			except subprocess.CalledProcessError as e:
				print e.output
				return 1
	return 0

def make_args(classes, values):
	return " ".join([make_flag(c, v) for c, v in zip(classes, values)])

def make_flag(c, v):
	return "-" + c + " " + v


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('corpus', type=str, help="Path to the corpus")
	parser.add_argument('config', type=str, help="Path to the config file in format required by openSMILE")
	parser.add_argument('-f', '--features', type=str, help="Path to the csv file containing classes")
	parser.add_argument('-o', '--output', type=str, default="output.arff", help="Path to the output arff file")
	parser.add_argument('--arfftargets', type=str, default="arff_targets_gui_labels.conf.inc", help="Path to the config file with the labels")
	args = parser.parse_args()
	ret = extract(args)
	exit(ret)

