import json


# 1. nama fungsi -> calculator
# 2. Desc -> panggil description ini kalau kamu mau mengunakan calculator untuk operasi matematika
# 3. Parameters -> parameter apa yang harus lu isi

# def penjumlahan(a, b):
#     return a + b

# 1. penjumlahan
# 2. Ambil ini kalau mau menjumlahkan
# 3. a, b

# penjumlahan(20, 20)

def add(a, b):
    return a + b

add(5)


basic_template = {
    'type': 'function',
    'function': {
        'name': 'function_name',
        'description': 'apa yang function lakukan',
        'parameters': {
            'type': 'object',
            'properties': {
                'param1':{
                    'type':'string',
                    'description':'penjelasan parameter'
                },
                'param2':{
                    'type':'string',
                    'description':'penjelasan parameter'
                }
            },
            'required':['param1']
        }
    }
}

'''
5 PARAMETER UTAMA:
1. STRING
2. NUMBER
3. BOOLEAN
4. ARRAY
5. OBJECT
'''

basic_template = {
    'type': 'function',
    'function': {
        'name': 'function_name',
        'description': 'apa yang function lakukan',
        'parameters': {
            'type': 'object',
            'properties': {
                'param1':{
                    'type':'string',
                    'description':'penjelasan parameter'
                },
                'param2':{
                    'type':'number',
                    'description':'penjelasan parameter'
                },
                'param3':{
                    'type':'boolean',
                    'description':'penjelasan parameter'
                },
                'param4':{
                    'type':'array',
                    'items': {'type':'string'}, #bisa diganti typenya
                    'description':'penjelasan parameter'
                },
                'param5':{
                    'type':'object',
                    'properties': {
                        'city':{'type':'string'},
                        'zip_Code':{'type':'string'},
                    }, #bisa diganti typenya
                    'description':'penjelasan parameter'
                }
            },
            'required':['param1']
        }
    }
}