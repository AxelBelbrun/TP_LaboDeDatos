import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt
import scipy
import numpy
import seaborn as sns


#%%
localidades = pd.read_csv('/home/clinux01/Descargas/localidades-censales.csv')
clae2 = pd.read_csv("/home/clinux01/Descargas/diccionario_clae2.csv")
cod_departamento = pd.read_csv("/home/clinux01/Descargas/diccionario_cod_depto.csv")
median = pd.read_csv("/home/clinux01/Descargas/w_median_depto_priv_clae2.csv")
padron = pd.read_csv("//home/clinux01/Descargas/padron-de-operadores-organicos-certificados.csv", encoding="windows-1252")

codigos_distintos = sql^"""SELECT *
                 FROM cod_departamento
                 INNER JOIN localidades
                 ON id_provincia_indec = provincia_id AND
                 nombre_departamento_indec = departamento_nombre
                 WHERE codigo_departamento_indec != departamento_id """
nombres_distintos = sql^"""SELECT *
                 FROM cod_departamento
                 INNER JOIN localidades
                 ON id_provincia_indec = provincia_id AND
                 codigo_departamento_indec = departamento_id
                 WHERE nombre_departamento_indec != departamento_nombre """

#Cambiamos los nombres y códigos que difieren, optamos por tomar de referencia los de la tabla localidades
nombres_a_cambiar = {'Chascomús/Lezama':'Chascomús' ,'Ezeiza' : 'José M. Ezeiza', 'General San Martín' : 'Ciudad Libertador San Martín',
'General Juan Facundo Quiroga': 'General Juan F. Quiroga',
'Ángel Vicente Peñaloza': 'General Ángel V. Peñaloza',
'Constitución': 'Villa Constitución',
'Juan Felipe Ibarra': 'Juan F. Ibarra',
'Ezeiza': 'José M. Ezeiza', "CABA" : "CIUDAD AUTONOMA BUENOS AIRES" }

cod_departamento.loc[:,["nombre_departamento_indec"]] = cod_departamento.nombre_departamento_indec.replace(nombres_a_cambiar) 
cod_departamento.loc[:,["codigo_departamento_indec"]] = cod_departamento.codigo_departamento_indec.replace({94014 : 94015}) 

#Creamos la tabla de provincias
provincias = sql ^ """
                 SELECT DISTINCT id_provincia_indec AS prov_id, nombre_provincia_indec AS prov_nombre
                 FROM cod_departamento
                 ORDER BY prov_id
                """
#--------------------------------------------------------------
#Eliminamos primero las columnas "función" y "fuente" que no aportan nada:
localidades = localidades.drop(['funcion'], axis = 1) 
localidades = localidades.drop(['fuente'], axis = 1)
#Eliminamos ahora las filas que puedan tener NULLs:
localidades = localidades.dropna()
#Creamos un nuevo DataFrame sin las columnas que vamos a trasladar a otras tablas:
localidades = sql^"""SELECT  categoria,centroide_lat,centroide_lon, departamento_id, departamento_nombre, municipio_id,
                               municipio_nombre, id AS localidad_id, nombre AS localidad_nombre
                               FROM localidades"""
#Ponemos todos los nombres en mayúsculas
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id,
                            UPPER(departamento_nombre) AS departamento_nombre,
                            municipio_id, UPPER(municipio_nombre) AS municipio_nombre,
                            localidad_id, UPPER(localidad_nombre) AS localidad_nombre, 
                            FROM localidades """
#Eliminamos las tildes de los departamentos para unificar los valores con las demás tablas
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id, 
                            REPLACE(departamento_nombre,'Á','A') AS departamento_nombre,municipio_id,
                            municipio_nombre, localidad_id,REPLACE(localidad_nombre,'Á','A')  AS localidad_nombre
                            FROM localidades"""
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id, 
                            REPLACE(departamento_nombre,'É','E') AS departamento_nombre,municipio_id,
                            municipio_nombre, localidad_id,REPLACE(localidad_nombre,'E','E')  AS localidad_nombre
                            FROM localidades"""
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id, 
                            REPLACE(departamento_nombre,'Í','I') AS departamento_nombre,municipio_id,
                            municipio_nombre, localidad_id,REPLACE(localidad_nombre,'Í','I')  AS localidad_nombre
                            FROM localidades"""
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id, 
                            REPLACE(departamento_nombre,'Ó','O') AS departamento_nombre,municipio_id,
                            municipio_nombre, localidad_id,REPLACE(localidad_nombre,'Ó','O')  AS localidad_nombre
                            FROM localidades"""
localidades = sql^"""SELECT categoria,centroide_lat,centroide_lon,departamento_id, 
                            REPLACE(departamento_nombre,'Ú','U') AS departamento_nombre,municipio_id,
                            municipio_nombre, localidad_id,REPLACE(localidad_nombre,'Ú','U')  AS localidad_nombre
                            FROM localidades"""   
#--------------------------------------------------------------
municipios = sql^"""SELECT DISTINCT municipio_id, municipio_nombre, departamento_id
                    FROM localidades"""  
#--------------------------------------------------------------                            
#Repetimos el procedimiento hecho para localidades
municipios.to_csv('municipios.csv')
municipios = sql^ """ SELECT municipio_id, REPLACE(municipio_nombre,'Á','A') AS municipio_nombre, departamento_id
                                FROM municipios"""
municipios = sql^ """ SELECT municipio_id, REPLACE(municipio_nombre,'É','E') AS municipio_nombre, departamento_id
                                FROM municipios"""
municipios = sql^ """ SELECT municipio_id, REPLACE(municipio_nombre,'Í','I') AS municipio_nombre, departamento_id
                                FROM municipios"""
municipios = sql^ """ SELECT municipio_id, REPLACE(municipio_nombre,'Ó','O') AS municipio_nombre, departamento_id
                                FROM municipios"""
municipios = sql^ """ SELECT municipio_id, REPLACE(municipio_nombre,'Ú','U') AS municipio_nombre, departamento_id
                                FROM municipios"""
localidades.to_csv('localidades.csv')

#%%


#%%
#Renombramos 'codigo_departamento_indec' como departamento_id y 'nombre_departamento_indec' como departamento_nombre para que tengan
#nombres mas cortos y declarativos, también ponemos todo en mayúsculas y sin tildes como arriba
departamentos = sql ^  """
                 SELECT codigo_departamento_indec AS departamento_id, nombre_departamento_indec AS departamento_nombre,
                 id_provincia_indec AS prov_id
                 FROM cod_departamento
                """
#Ponemos todos los nombres en mayúsculas
departamentos = sql^"""SELECT departamento_id, UPPER(departamento_nombre) AS departamento_nombre, prov_id
                            FROM departamentos """

#Eliminamos las tildes en los deptos para unificar valores con las demás tablas
departamentos = sql^"""SELECT departamento_id,REPLACE(departamento_nombre,'Á','A') AS departamento_nombre,
                            prov_id
                            FROM departamentos"""
departamentos = sql^"""SELECT departamento_id, REPLACE(departamento_nombre,'É','E') AS departamento_nombre,
                            prov_id
                            FROM departamentos""" 
departamentos = sql^"""SELECT departamento_id, REPLACE(departamento_nombre,'Í','I') AS departamento_nombre,
                            prov_id
                            FROM departamentos"""
departamentos = sql^"""SELECT departamento_id, REPLACE(departamento_nombre,'Ó','O') AS departamento_nombre,
                            prov_id
                            FROM departamentos"""
departamentos = sql^"""SELECT departamento_id, REPLACE(departamento_nombre,'Ú','U') AS departamento_nombre,
                            prov_id
                            FROM departamentos"""                            
#--------------------------------------------------------------  
departamentos.to_csv('departamentos.csv', index = False)

#%%

#Creamos la tabla que contiene todas las ubicaciones con toda la info, será útil luego                
prov_deptos = sql^"""SELECT * FROM provincias
                            INNER JOIN departamentos
                            ON departamentos.prov_id = provincias.prov_id
                            """
prov_deptos_muni = sql^"""SELECT * FROM prov_deptos
                INNER JOIN municipios
           
                ON prov_deptos.departamento_id = municipios.departamento_id"""
ubicaciones_completo = sql^"""SELECT * FROM prov_deptos_muni
                INNER JOIN localidades
           
                ON prov_deptos_muni.municipio_id =  localidades.municipio_id"""
#%%
"""Ahora vamos a borrar las columnas "pais_id", "localidad" y "pais" del padrón 
porque no nos aportan información relevante. Además borramos la columna "provincia",
porque con el Prov_id es suficiente y evitamos así redundancia""" 

padron=padron.drop(['pais_id'], axis = 1)
padron=padron.drop(['pais'], axis = 1)
padron=padron.drop(['provincia'], axis = 1)

padron=padron.drop(['localidad'], axis = 1)

#Creamos la relación externa "certificadores" para relacionar id's y nombres
certificadores = sql^"""
                    SELECT DISTINCT Certificadora_id AS certificadora_id, certificadora_deno AS 
                    certificadora_nombre
                    FROM padron
                    """

#Hacemos algo similar para las categorías
categorias = sql^"""
                    SELECT DISTINCT categoria_id, categoria_desc AS categoria_nombre
                    FROM padron
                    """
"""Ahora quitamos los "nombres" de los certificadores y categorías, ya que están en una tabla aparte,
y nos quedamos solo con los id's correspondientes"""
padron=padron.drop(['certificadora_deno'], axis = 1)
padron=padron.drop(['categoria_desc'], axis = 1)

#Renombramos la columna razón social quitándole la tilde a la o.
padron.rename(columns = {'razón social':'razon_social'}, inplace = True)
#Ahora, definimos que todas los establecimientos con valor "NC" serán llamados "Establecimiento_unico" 
padron.loc[:,["establecimiento"]] = padron.establecimiento.replace({"NC" : "Establecimiento_unico"})
#Creamos nueva tabla para relacionar establecimientos y los productos que venden
productos=sql^"""
                    SELECT DISTINCT razon_social,establecimiento,productos
                    FROM padron

                    ORDER BY razon_social,establecimiento
                    """
productos["productos"]=productos["productos"].str.split(", ")
productos= productos.explode("productos").reset_index(drop=True)
productos["productos"]=productos["productos"].str.split(" Y ")
productos= productos.explode("productos").reset_index(drop=True)
productos["productos"]=productos["productos"].str.split("+")
productos= productos.explode("productos").reset_index(drop=True)
padron=padron.drop(['productos'], axis = 1)




#Corregimos un error de ortografía 
padron = sql^"""
                    SELECT DISTINCT provincia_id AS prov_id, 
                    departamento,REPLACE(rubro, 'AGICULTURA', 'AGRICULTURA') as rubro,
                                    categoria_id,Certificadora_id,razon_social,establecimiento
                    FROM padron
                    ORDER BY departamento,rubro,categoria_id,Certificadora_id,razon_social
                    """ 

"""Ahora, dado que en la tabla del padrón hay inconsistencias en la columna de departamentos (hay nombres de departamentos,
ciudades o localidades), vamos a solucionar eso haciendo INNER JOIN de esta columna con las tablas de 
departamentos, localidades y municipios, y en caso de que haya localidades y municipios que concuerden, las reemplazaremos
por sus correspondientes departamentos"""

son_departamentos = sql^""" SELECT padron.prov_id, departamento, rubro, categoria_id,
                            Certificadora_id, razon_social,establecimiento
                            FROM padron
                            INNER JOIN departamentos
                            ON departamento = departamentos.departamento_nombre AND
                            departamentos.prov_id = padron.prov_id """
padron_sin_departamentos = sql^"""SELECT * FROM padron
                            EXCEPT
                            SELECT * FROM son_departamentos """


                                            
son_municipios = sql^ """ SELECT padron_sin_departamentos.prov_id, departamento, rubro, categoria_id,
                            Certificadora_id, razon_social,establecimiento
                            FROM padron_sin_departamentos
                            INNER JOIN ubicaciones_completo
                            ON departamento = ubicaciones_completo.municipio_nombre AND 
                            padron_sin_departamentos.prov_id = ubicaciones_completo.prov_id
                            """ 
padron_sin_municipios =  sql^"""SELECT * FROM padron_sin_departamentos
                            EXCEPT
                            SELECT * FROM son_municipios """
son_localidades = sql^ """ SELECT padron_sin_municipios.prov_id, departamento, 
                            rubro, categoria_id,
                            Certificadora_id, razon_social,establecimiento
                            FROM padron_sin_municipios
                            INNER JOIN ubicaciones_completo
                            ON departamento = ubicaciones_completo.localidad_nombre AND 
                            padron_sin_municipios.prov_id = ubicaciones_completo.prov_id
                            """ 
padron_sin_localidades = sql^"""SELECT * FROM padron_sin_municipios
                            EXCEPT
                            SELECT * FROM son_localidades """

"""Ahora que tenemos en claro cuáles valores de la columna "departamento" son departamentos,
cuáles son "municipios" y cuáles "localidades", podemos, en los casos que corresponda, asignar los
departamentos que no estaban"""

padron_final_1 = sql^"""SELECT *
                FROM padron_sin_municipios
                INNER JOIN localidades
                ON departamento = localidades.localidad_nombre"""
padron_final_1 = sql^"""SELECT prov_id, departamento, rubro,
                categoria_id, Certificadora_id, razon_social,establecimiento
                FROM padron_final_1"""
padron_final_2 = sql^"""SELECT *
                FROM padron_sin_departamentos
                INNER JOIN municipios
                ON departamento = municipios.municipio_nombre"""
padron_final_2 = sql^"""SELECT prov_id, departamento, rubro,
                categoria_id, Certificadora_id, razon_social,establecimiento
                FROM padron_final_2"""                
padron = sql^"""SELECT * FROM padron_final_1
                      UNION
                      SELECT * FROM padron_final_2
                      UNION
                      SELECT* FROM son_departamentos"""
                      
#ACLARACIÓN: Luego de este proceso quedan tuplas en las cuales su "departamento" no corresponde a ninguna localidad,
#municipio o departamento de los registros disponibles, por lo que quedan fuera del padrón. Un ejemplo es "ALMANZA". 


#%%

#Renombramos las columnas de 'codigo_departamento_indec', 'id_provincia_indec', 'clae2' y 'w_median' para que 
#tengan nombres declarativos
#Eliminamos la columna 'id_provincia_indec' pues ya tenemos la relacion prov_departamento
#Eliminamos los registros que tengan NULLS en el departamento y salario mediano<0
#ACLARAR QUE ASUMIMOS O INTERPRETAMOS QUE GANADERIA(ACTIVIDAD CON MAS OPERADORES) ESTA INCLUIDA EN RUBRO_id=1
salarios = sql ^  """
                 SELECT DISTINCT fecha,codigo_departamento_indec AS departamento_id, clae2 AS rubro_id,
                 w_median AS salario_mediano
                 FROM median
                 WHERE salario_mediano>0 AND departamento_id IS NOT NULL
                 ORDER BY fecha, departamento_id, rubro_id, salario_mediano
                """



#Creamos una tabla que es igual a la de salarios original, salvo que en vez
#del id de la provincia tiene el nombre. Esto nos va a servir para luego hacer
#comparaciones y filtrar por provincia de una manera más fácil de entender
salarios = sql ^ """
                 SELECT DISTINCT fecha,s.departamento_id, prov_nombre,p.prov_id,rubro_id,salario_mediano
                 FROM salarios AS s, departamentos AS d, provincias AS p
                 WHERE d.departamento_id=s.departamento_id AND p.prov_id=d.prov_id
                 ORDER BY fecha,s.departamento_id,prov_nombre,rubro_id,salario_mediano
                 
                """
salarios_dic_2022 = salarios[salarios['fecha'] == '2022-12-01'] 
#Esta funcion en cada iteracion del i reemplaza el valor del año en un mes especifico (ej: 2014-01-01 0 2014-02-01) con el año respectivo
#En cada iteracion de j aumenta el año en 1 (j=0, año=2014; j=1, año=2015)
#2023 lo hicimos aparte ya que no hay un registro completo del año, solo hay de enero y febrero
for j in range(9):
    año=str(2014+j)
    for i in range(1,13):
        mes=str(i)
        if i<10:
            salarios['fecha']=salarios['fecha'].replace([año+'-0'+mes+'-01'],año)
        else:
            salarios['fecha']=salarios['fecha'].replace([año+'-'+mes+'-01'],año)
salarios['fecha']=salarios['fecha'].replace(['2023-01-01'],'2023')
salarios['fecha']=salarios['fecha'].replace(['2023-02-01'],'2023')
               
promedio_anual = sql^"""
                    SELECT fecha AS año, AVG(salario_mediano) AS promedio_anual
                    FROM salarios
                    GROUP BY fecha
                    ORDER BY fecha
"""
promedio_anual_total = promedio_anual['promedio_anual'].mean()
desvio = promedio_anual['promedio_anual'].std()           


promedio_por_provincia = sql^"""
                    SELECT fecha AS año, prov_nombre, AVG(salario_mediano) AS promedio
                    FROM salarios
                    GROUP BY fecha,prov_nombre
                    ORDER BY fecha,prov_nombre
"""

#Punto i, sección v:
nac_tot = promedio_anual
promedio_salario = nac_tot['promedio_anual'].median()
x = nac_tot['promedio_anual'].std()
nac_tot['promedio_anual'].plot(kind = 'kde')

plt.axvline(nac_tot['promedio_anual'].mean(), color = 'orange')#la media
#naranja

plt.axvline(nac_tot['promedio_anual'].median() , color = 'red')#mediana
#rojafrom scikit-learn import 

plt.axvline(nac_tot['promedio_anual'].quantile(0.5) , color = 'blue' , linestyle = '--')
#cuantil 50, en azul
plt.axvline(nac_tot['promedio_anual'].quantile(0.25) , color = 'green')
#cuartil 25 en verde 
plt.axvline(nac_tot['promedio_anual'].quantile(0.75) , color = 'violet')
#cuartil 75 en violeta
plt.axvline(nac_tot['promedio_anual'].mean()+nac_tot['promedio_anual'].std(), color='black' , linestyle='-.')

plt.axvline(nac_tot['promedio_anual'].mean()-nac_tot['promedio_anual'].std(), color='black' , linestyle='-.')

plt.show()
plt.close()




#Consultas punto a)

provincias_sin_operadores = sql^"""
                 SELECT DISTINCT p.prov_id
                 FROM provincias as p
                 EXCEPT
                 SELECT DISTINCT pa.prov_id
                 FROM padron as pa
                """
provincias_sin_operadores1 = sql^"""
                 SELECT DISTINCT p.prov_id, pr.prov_nombre as provincia
                 FROM provincias_sin_operadores AS p
                 INNER JOIN provincias as pr
                 ON pr.prov_id=p.prov_id
                """

#Consultas punto b)
departamentos_de_padron =sql^"""
                    SELECT DISTINCT departamento
                    FROM padron
                    """
departamentos_sin_operadores = sql^"""
                 SELECT DISTINCT UPPER(departamento_nombre) as departamentos
                 FROM departamentos
                 EXCEPT
                 SELECT DISTINCT departamento
                 FROM padron
                """

#Consultas punto c)
#ACLARAR QUE NO SEPARAMOS LOS RUBROS MIXTOS YA QUE NO VARIARIA MUCHO
ranking_operadores = sql^"""
                 SELECT DISTINCT rubro,COUNT() as cantidad
                 FROM padron
                 GROUP BY rubro
                 ORDER BY cantidad DESC
                """


actividad_con_mas_operadores =sql^"""
                 SELECT DISTINCT r1.rubro, r1.cantidad
                 FROM ranking_operadores as r1
                 WHERE r1.cantidad=(
                 SELECT MAX(r2.cantidad)
                 FROM ranking_operadores as r2
                 )
                """

#Consultas punto d)


#Sacamos el promedio anual de agricultura y lo ordenamos del mas reciente al mas antiguo

fruticultura_promedio_anual= sql^"""
                 SELECT DISTINCT fecha,AVG(salario_mediano) as promedio
                 FROM salarios
                 WHERE fecha LIKE '2022%' AND rubro_id=1
                 GROUP BY fecha
                 ORDER BY fecha DESC
                """

#Graficos punto a)
operadores_por_provincia= sql^"""
                 SELECT DISTINCT prov_nombre as provincia,COUNT() as operadores
                 FROM padron as p
                 INNER JOIN provincias as pr
                 ON p.prov_id=pr.prov_id
                 GROUP BY prov_nombre
                 ORDER BY prov_nombre ASC
                """

#uso scatterplot para graficar
sns.scatterplot(data=operadores_por_provincia, y="provincia", x="operadores")
plt.show()
plt.close()

#Graficos punto b)


boceto = sql^"""
                 SELECT razon_social, COUNT() as cantidad
                 FROM productos
                 GROUP BY razon_social
                 ORDER BY razon_social
                """

boceto2 =sql^"""
                 SELECT prov_nombre,p.razon_social,cantidad
                 FROM padron as p,boceto as b, provincias as pr
                 WHERE p.razon_social=b.razon_social AND pr.prov_id=p.prov_id
                 GROUP BY prov_nombre,p.razon_social,cantidad
                 ORDER BY prov_nombre,p.razon_social,cantidad
                """

boceto3 =sql^"""
                 SELECT prov_id ,cantidad
                 FROM boceto2
                 INNER JOIN provincias
                 ON boceto2.prov_nombre = provincias.prov_nombre
                """



sns.boxplot(data=boceto3, x='prov_id',y='cantidad').set(title ='Cantidad de Operadores por provincia')

plt.show()
plt.close()


rubro_claes = pd.read_csv('/home/clinux01/Descargas/rubro_claes.csv')
#Graficos punto c)

operadores_por_provincia = sql^""" SELECT prov_id, clae_2, COUNT() AS cantidad
                                    FROM padron
                                    INNER JOIN rubro_claes
                                    ON padron.rubro = rubro_claes.rubro
                                    GROUP BY prov_id,clae_2
                                    ORDER BY prov_id ,clae_2"""
salarios_prov_clae = sql^""" SELECT prov_id, rubro_id, AVG(salario_mediano) AS promedio
                             FROM salarios_dic_2022
                             GROUP BY prov_id, rubro_id
                             ORDER BY prov_id, rubro_id """
ejercicio_c = sql ^""" SELECT promedio, cantidad, clae_2, 
                       FROM salarios_prov_clae as a
                       INNER JOIN operadores_por_provincia as b
                       ON a.prov_id = b.prov_id AND b.clae_2 = a.rubro_id """
                  
print(sql^"""SELECT DISTINCT clae_2 FROM ejercicio_c""")                                              
x = ejercicio_c['promedio']

y = ejercicio_c['cantidad']

sns.scatterplot(data = ejercicio_c, x = 'promedio', y = 'cantidad', hue='clae_2',palette='deep')                                    
plt.show()
plt.close
#punto j, iv:
salarios2023 = salarios[salarios['fecha'].str.startswith('2023')]
salario_promedio_2023 = sql ^  """
                 SELECT DISTINCT prov_id,rubro_id, AVG(salario_mediano) AS promedio2023
                 FROM salarios2023
                 GROUP BY prov_id,rubro_id
                 ORDER BY prov_id,rubro_id
                """
sns.violinplot(data=salario_promedio_2023,x="prov_id",y="promedio2023").set(title ='Salarios promedio por provincia')

operadores_por_depto = sql^""" SELECT departamento,prov_id, COUNT() AS cantidad
                                    FROM padron
                                    GROUP BY departamento,prov_id
                                    ORDER BY departamento,prov_id """
operadores_por_depto = sql^"""SELECT departamento, departamento_id, a.prov_id, cantidad
                              FROM operadores_por_depto as a
                              INNER JOIN departamentos as b
                              ON departamento = departamento_nombre AND a.prov_id = b.prov_id """

                                
salario_promedio_depto = sql^""" SELECT departamento_id, prov_id, AVG(salario_mediano) AS promedio_2022
                                 FROM salarios
                                 WHERE fecha = 2022
                                 GROUP BY departamento_id, prov_id
                                 ORDER BY departamento_id, prov_id"""
relacion = sql^"""SELECT departamento, cantidad, promedio_2022
                  FROM salario_promedio_depto as a  
                  INNER JOIN operadores_por_depto as b
                  ON a.departamento_id =b.departamento_id"""
sns.scatterplot(data = relacion, x = 'promedio_2022' , y = 'cantidad')
