input_filename = "D:\\OneDrive - The University Of Hong Kong\\SPP\\RnD\\1\\2024-03-15-10-05-31_position.txt"
object_id_Dic = {} # Object ID: 3D position
output_filename = input_filename.split("_position")[0] + "_position_new.txt"

with open(input_filename, 'r') as input_file, open(output_filename, "w") as output_file:
    tmp_list = ['x','y','z','videoname','frameid','objid','left','top','bbox_w','bbox_h','a1','a2','a3','a4','objname','b1']
    output_file.write('\t'.join(tmp_list))
    output_file.write('\n')
    data = input_file.readlines()

    for line in data:
        elements = line.split(' ')
        if elements != ['\n']:
            if len(elements) == 17:
            	elements[-3:-1] = ['_'.join(map(str, elements[-3:-1]))]
            elif len(elements) == 18:
            	print(elements)
            	elements[-4:-1] = ['_'.join(map(str, elements[-4:-1]))]
            tmp_list.extend(elements)
            output_file.write('\t'.join(elements))