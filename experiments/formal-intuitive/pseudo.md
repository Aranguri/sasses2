lista = lista de teoremas
puntueador = Puntuador() # no confundir con puteador :)

def demostrar(qvq): # pasamos como parametro el teorema que queremos demostrar
	fiteados = [fitear(qvq, t) for t in lista]
	doble_fiteados = [doble_fitear(qvq, t) for t in lista]

	if any(fiteados):
	    return True # si nuestro qvq fitea en algún v, ganamos.
	elif any(doble_fiteados):
	    teoremas = [p for p=>q in doble_fiteados]
	    demostrado = False
	    while not demostrado and len(teoremas) != 0:   
	        teorema = mejor_teorema(teoremas)
		teoremas.remove(teorema)
		demostrado = demostrar(teorema)
	    return demostrado
	else:
	    return False

def mejor_teorema(teoremas):
	puntajes = {t: 0 for t in teoremas}
	for t1, t2 in pares(teoremas):
		resultado = caja_negra(t1, t2)
		if resultado == 0: puntajes[t1] += 1
		elif resultado == 1: puntajes[t2] += 1
		# el resultado también puede ser 2 (ie un empate)

def caja_negra(t1, t2):
	# we need a short-term memory.
	# Pasamos los teoremas de texto a un vector en (100, 8) dimensiones.
	#  El 100 viene de que usamos teoremas de hasta 100 caracteres.
	#  El 8 viene de que usamos 8 caracteres diferentes.
	t1, t2 = representar(t1), representar(t2)
 	# Transformamos ambos teoremas a otro espacio para hacerlos más compactos y
	#  más fáciles de manejar. (Usamos la misma transformación para ambos.)
	t1, t2 = transformar(t1), transformar(t2)
	input = concatenar(t1, t2)
	return puntuador.puntuar(input)

# Otras funciones
asignación(t) = return una asiganción arbitraria de formulas para las fariables de t.
fitear(t1, t2) = return True if existe una asignación tal que asignación(t2) = t1 else False
doble_fitear(t1, t2) = return asignación(t2) if existe un par de asignaciones tal que asignación(t2) = asignación(t1) else False
# Para las implementaciones de fitear y doble_fitear, creo que se pueden hacer comparando los arboles de ambos teoremas.
uno_de_k(char) = return un array de 0s con un 1 en la posición dada por chars[char]
representar(t) = return [uno_de_k(char) for char in t]

def transformar(t):
	t = [np.dot(W, char) for char in chars] # Convertimos (100, 8) a (100, 2)
	t = t.flatten() # Convertimos (100, 2) a (200,)
	t = np.dot(U, t) # Convertimos (200,) a (50,)
	return t

class Puntuador:
	def __init__(self):
		self.state = empty
		self.weights = random weights

	def puntuar(self, input):
		# Esta es una abstración de una red neuronal con
		#  loops.
		W, U, V = self.weights
		self.state = np.dot(W, input) + np.dot(U, self.state)
		return np.dot(V, self.state)



# Other things to think: formulas embedding, simpler model: one without double-fit




	
