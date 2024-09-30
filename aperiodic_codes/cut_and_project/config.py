import os
prefix = '../../data/apc'
executable = './cut_multi'

tests = {
    "0": {
        "thetas": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "offset": [0, 0, 0, 0, 0, 0],
        "code1": [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1],
        "code2": [1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0]
    },
    "1027805": {
        "thetas": [6.406953430308097, 5.349274542356475, 0.6822133844997569, 0.5974713336012304, 1.3252537411849554, 0.912736410967225, 3.8448421914141266, 5.893658087392231, 1.3755798562976995, 1.4273007225028496, 2.0568879800919, 6.228337204298493, 2.4722791967308706, 1.5754352568017014, 0.4294626881179672],
        "offset": [0.71328045, 0.79239787, 0.51035573, 0.33944886, 0.663739, 0.8410469]
    },
    # 1
    "3356642": {
        "thetas": [0.7925399352064098, 5.86165748971021, 0.03977415560355426, 0.37057069475667836, 0.8807606851562947, 5.027967501543387, 5.8204916494225944, 0.6870210651653699, 6.235847319912712, 1.2937761277503594, 0.5676246016977526, 3.2171720855548784, 3.2000377450032462, 4.03885444310909, 2.398455176762991],
        "offset": [0.16825907, 0.00843047, 0.4987704, 0.05320437, 0.46910073, 0.33991224]
    },
    # 2
    "468936": {
        "thetas": [1.5827105265733632, 0.03531052852250263, 4.540416315828664, 2.5351635744644074, 3.0595231491367763, 0.6443398499490571, 0.4653690443301789, 4.3122098360648256, 3.6858283210811074, 4.277042982205427, 1.0994844429932844, 5.478900676536209, 2.8150658335196717, 1.6659192359069779, 2.8070690932568385],
        "offset": [0.45449589, 0.8681517, 0.45948883, 0.08294291, 0.35848052, 0.54544266]
    },
    # 3
    "20240903_n=3_1": {
        "thetas": [2.594869346392367,0.9477109111037187,3.620412204482349,5.283264728923087,3.6816833892843457,-0.5959013298160938,4.382704240484682,0.09458794430181908,4.836072911632601,3.3235372560725627,1.0035735979880425,3.1895322061975233,2.793452898208629,0.3547799862139244,2.774503370428157],
        "offset": [0.29806990830179625,0.12993246148702742,0.574709309736253,0.39003543441180044,0.13260187449328498,0.6966313017934023]
    },
    # 4
    "20240903_n=3_2": {
        "thetas": [-0.6486077234956892,2.788739850710069,4.2161856631295285,3.9835509703418115,2.007054752362599,1.635616042577092,2.0951845657801185,5.4853656517130736,4.058483023261345,0.26204978523296796,3.3945252945776163,2.9377537398296716,2.845914141772035,5.521586454424707,1.7252598170438158],
        "offset": [0.18526040488890427,0.39674112322205346,0.6246256057585734,0.5171191359691727,0.5630026241574658,0.9968372945829661]
    },
    # 5
    "20240903_n=3_3": {
        "thetas": [2.565778499785135,5.789588365836854,2.1818116163908807,5.572566734091489,5.756462276489271,3.2684456297203175,1.5650208356860904,5.64620931274392,4.380914310424174,3.693033318165752,6.549915181005626,0.21146371608192888,1.5678733998604337,0.9572206527002634,3.6917623790750413],
        "offset": [0.8806132621551196,0.48623182053847547,0.1025221646357134,0.23087111967042573,0.5023800018452489,0.8659021972421654]
    },
    # 6
    "20240903_n=4_1": {
        "thetas": [2.594869346392367,0.9477109111037187,3.620412204482349,5.283264728923087,3.6816833892843457,-0.5959013298160938,4.382704240484682,0.09458794430181908,4.836072911632601,3.3235372560725627,1.0035735979880425,3.1895322061975233,2.793452898208629,0.3547799862139244,2.774503370428157],
        "offset": [0.29806990830179625,0.12993246148702742,0.574709309736253,0.39003543441180044,0.13260187449328498,0.6966313017934023]
    },
    # 7
    "20240903_n=4_2": {
        "thetas": [-0.6486077234956892,2.788739850710069,4.2161856631295285,3.9835509703418115,2.007054752362599,1.635616042577092,2.0951845657801185,5.4853656517130736,4.058483023261345,0.26204978523296796,3.3945252945776163,2.9377537398296716,2.845914141772035,5.521586454424707,1.7252598170438158],
        "offset": [0.18526040488890427,0.39674112322205346,0.6246256057585734,0.5171191359691727,0.5630026241574658,0.9968372945829661]
    },
    # 8
    "20240903_n=4_3": {
        "thetas": [2.565778499785135,5.789588365836854,2.1818116163908807,5.572566734091489,5.756462276489271,3.2684456297203175,1.5650208356860904,5.64620931274392,4.380914310424174,3.693033318165752,6.549915181005626,0.21146371608192888,1.5678733998604337,0.9572206527002634,3.6917623790750413],
        "offset": [0.8806132621551196,0.48623182053847547,0.1025221646357134,0.23087111967042573,0.5023800018452489,0.8659021972421654]
    },
    "20240914_n=3_DIRS27_1": {  # high n_low, 0 n_anti
        "code1": [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1],
        "code2": [1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
        "thetas": [-0.07489847573567546,-0.060778299220810586,-0.35076282492043276,0.22781160728998012,-0.16134201679627627,-0.10670959636699755,0.44384384200552596,0.02505303193656889,0.20923456178078512,0.3374705528977506,-0.1426706139116105,-0.07878380769240524,-0.3629747961460306,-0.041277182413093465,-0.24294613460770065],
        "offset": [0.7782220691664274,0.6823616837537663,0.8564518017108291,0.8991781454342163,0.8035935441522366,0.6188969031890882]
    },
    "20240914_n=3_DIRS27_2": {  # high n_low, 0 n_anti
        "code1": [1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,0,0],
        "code2": [1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
        "thetas": [0.06597381579632328,-0.24446143129529305,-0.4477586300412575,-0.18706258332512277,0.3115327089265842,0.3430473958820039,-0.18056085679753478,-0.027437603044428063,0.2856422950652892,-0.20664323459337258,0.006907082482093285,0.06599495543485431,0.23522802540830134,0.6433278489192991,0.20261042154375383],
        "offset": [0.033168238758718105,0.9479742572114487,0.9582955415865901,0.6806446527080967,0.6281908737252574,0.5737755153260353]
    },
    "20240920_n=3_DIRS27_1": {
        "code1": [1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0],
        "code2": [1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1],
        "thetas": [0.11053839983583207,0.10445826427718782,-0.20046509326013928,-0.05117206824295077,0.10870997767538855,0.06750544854207115,-0.06071210865728064,-0.09978930872975642,0.34715701209309335,-0.013132618604160226,-0.19919356275386846,-0.10628808857329994,-0.038987288066107136,0.004158886810534659,0.020581955735074618],
        "offset": [0.42324949540959167,0.6611369806715365,0.9902403368992637,0.8593731373165288,0.1177629369382649,0.8399150934556178]
    },
    "20240926_n=3_DIRS27_1": {
        "code1": [0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
        "code2": [1,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0],
        "thetas": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.008808024048145385],
        "offset": [0.9991169266871937,0.7812324742246252,0.5487436476523516,0.9722811384587086,0.8267619553000166,0.0576183043421475]
    },
    "20240926_n=3_DIRS27_2": {
        "code1": [0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
        "code2": [1,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0],
        "thetas": [-0.10182807667841269,0.08508134398483475,-0.1640542919338752,0.012142879411901238,0.10099873292580082,-0.10868112997874327,-0.0899202182784169,-0.05619038857562167,0.3636260257957463,-0.06885404422188254,-0.215765227633734,-0.16070199537292273,-0.1482008295115896,0.07496916728158909,-0.006816424440742351],
        "offset": [0.42324949540959167,0.6611369806715365,0.9902403368992637,0.8593731373165288,0.1177629369382649,0.8399150934556178]
    }
}