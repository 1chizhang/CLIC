import handle_util as UTIL
import os




tplt = "{0:^25}\t{1:^10}\t{2:^10}"


def read_file(file):
    # print(file)
    result = []
    data = open(file).readlines()
    # print(len(data))
    num_cnn=0
    sumNum = 0
    for i in range(len(data)):
        if "cnnflag = 1" in data[i]:
            num_cnn += 1
        if "cnnflag" in data[i]:
            sumNum += 1
        # if "Stream 0 PSNR " in data[i]:
        if "Total Frames" in data[i]:
            line_data0 = data[i-3].split()
            result.append(line_data0[12])
            line_data = data[i+1].split()
            for s in line_data:
                if UTIL.isNum(s) and s != '0':
                    result.append(s)
            #print(line_data)
            break
    # result.append(str(num_cnn/sumNum))
    return result


if __name__ == '__main__':
    path = r"\log"
    file_list = UTIL.iter_files(path)
    file_list = UTIL.sort_list(file_list)
    # print(tplt.format("file name", "QP", "bitrate", "Y", "U", "V",
    #                    "yuv", "num"))
    print(tplt.format("file name","bpp",
                      "yuv",))

        # print()
        # print()
        # print()
        # print(file_list)
        # print()
    for file in file_list:
        # print(file)
        # print(i)
        if file.split('.')[-1] == "txt" :

            result = read_file(file)
            # print(result)
            file_name = os.path.basename(file).split(".")[0]
            # print(tplt.format(file_name.split("_")[1], file_name.split("_")[0], result[1], result[2],
            #                   result[3], result[4], result[5], result[-1]))
            print(tplt.format(file_name,  int(result[0])/768/512,
                              result[6]))