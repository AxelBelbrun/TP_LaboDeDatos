import csv

def leer_parque(nombre_archivo, parque):
    f = open(nombre_archivo, encoding="utf-8")
    filas = csv.reader(f)
    data = []
    encabezado = next(filas)
    for fila in filas:
        registro = dict(zip(encabezado, fila))
        if registro['espacio_ve'] == parque:
            data.append(registro)
    return data


print(len(leer_parque('arbolado-en-espacios-verdes.csv', 'GENERAL PAZ')))

def especies(lista_arboles):
    lista_especies = []
    for arbol in range(len(lista_arboles)):
        lista_especies.append(lista_arboles[arbol]['nombre_com'])
    conjunto_especies = set(lista_especies)
    return conjunto_especies


lista_arboles = leer_parque('arbolado-en-espacios-verdes.csv', 'GENERAL PAZ')

print(especies(lista_arboles))
print(len(especies(lista_arboles)))

