import os

parent_path = os.getcwd()

time = 120

candle = [1, 2]
door = [0.050, 0.150]
vertical_opening_1 = ["open", "closed", "HVAC"]
vertical_opening_2 = ["open", "closed"]


for i in range(0,len(candle)):
    for j in range(0,len(door)):
        for k in range(0,len(vertical_opening_1)):
            for l in range(0,len(vertical_opening_2)):
                os.chdir(parent_path)
                working_folder = os.path.join(parent_path,"c{}_d{}_vod{}_voc{}".format(i,j,k,l))
                os.mkdir(working_folder)

                if vertical_opening_1[k] == "open":
                    with open('template.fds', 'r') as file :
                        template = file.read()
                    file.close()
                    template = template.replace('###VOD###', "&HOLE XB = 0.32,0.40, -0.03,0.03, 0.21,0.25, / vertical opening door".format(i,j,k,l))
                elif vertical_opening_1[k] == "closed":
                    with open('template.fds', 'r') as file :
                        template = file.read()
                    file.close()
                    template = template.replace('###VOD###', "")
                else:
                    with open('template_hvac.fds', 'r') as file :
                        template = file.read()
                    file.close()

                template = template.replace('###TIME###', "{}".format(time))

                template = template.replace('###CHID###', "c{}_d{}_vod{}_voc{}".format(i,j,k,l))
                template = template.replace('###TITLE###', "Candle(s):{}; Door width:{}; Vertical opening door:{}; Vertical opening candle:{}".format(candle[i],door[j],vertical_opening_1[k],vertical_opening_2[l]))

                if candle[i] == 1:
                    template = template.replace('###CANDLE 1###', "&VENT XB = 0.92,0.96, -0.02,0.02, 0.00,0.00, COLOR='RED', SURF_ID='BURNER' / Candle 1")
                    template = template.replace('###CANDLE 2###', "")
                else:
                    template = template.replace('###CANDLE 1###', "&VENT XB = 0.92,0.96, -0.02,0.02, 0.00,0.00, COLOR='RED', SURF_ID='BURNER' / Candle 1")
                    template = template.replace('###CANDLE 2###', "&VENT XB = 0.84,0.88, -0.02,0.02, 0.00,0.00, COLOR='RED', SURF_ID='BURNER' / Candle 2")


                template = template.replace('###DOOR###', "&HOLE XB = 0.25,0.29, -0.1,0.1, -0.01,{} / Door".format(door[j]))

                if vertical_opening_2[l] == "open":
                    template = template.replace('###VOC###', "&HOLE XB = 0.86,0.94, -0.03,0.03, 0.21,0.25, / vertical opening candle")
                else:
                    template = template.replace('###VOC###', "")


                file_name = "c{}_d{}_vod{}_voc{}.fds".format(i,j,k,l)
                fds_file = open('{}'.format(os.path.join(working_folder, file_name)), 'w+')
                fds_file.write(template)
                fds_file.close()

                copy_sbatch = "cp start_job.batch {}".format(working_folder)

                start_fds = "sbatch start_job.batch"

                os.system(copy_sbatch)
                os.chdir(working_folder)
                os.system(start_fds)
