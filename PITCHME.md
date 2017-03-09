#HSLIDE

# C
"Wiesz kto wymyslił Pythona? Holendrzy, wiec jest taki lewacki - niby wszystko za Ciebie robi, a nic sie nie da w nim zrobic. 

A C? wiesz kto C wymyslil? Amerykanie! I to jest jezyk!"
- Piotr Tempczyk

#HSLIDE

## Główne różnice

- kompilowany, nie interpretowany
- bloki kodu oznaczone `{ }`, a nie wcięciami
- każda linijka kończy się średnikiem `;`
- silnie typowany
- samodzielne zarządzanie pamięcią - alokowanie i zwalnianie pamięci
- brak fancy struktur (słowniki, listy) - wszystko jest edytowane żywcem na pamięci

#HSLIDE

## Hello World
### C:
```
#include <stdio.h>
int main() {
	printf("Hello World!");
	return 0;
}
```
## python
```
print "Hello World!"
```

#HSLIDE

## Silnie typowany

**python**
```
def min(a, b):
	if a < b:
		return a
	return b
```

**C**
```
int min(int a, int b) {
	if(a < b) return a;
	return b;
}
float min(float a, float b) {
	//
}
long long min(long long a, long long b) {
	//
}
```

#HSLIDE

## Brak fancy struktur 1/2

**python**
```
T = xrange(100)
```

**C**
```
int *T;
T = malloc(100 * sizeof(int));
for (int i = 0; i < 100; i ++) {
	T[i] = i;
}
// Reszta kodu
// Na koniec
free(T)
```

#HSLIDE

## Brak fancy struktur 2/2

**python**
```
X = np.random.normal(size=(10, 10))
s = sum(map(sum, X)))
```

**C**
```
int T[10][10];
for(int i = 0; i < 10; i++)
	for(int j = 0; j < 10; j++)
		T[i][j] = rand();

int s = 0;
for(int i = 0; i < 10; i++)
	for(int j = 0; j < 10; j++)
		sum += T[i][j];
```

#HSLIDE

# CUDA

#HSLIDE

# Rozkład wątków
![Watki](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)

#HSLIDE

# Rozkład pamięci
![Pamięć](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/memory-hierarchy.png)

#HSLIDE
![DeviceQuery](devicequery.png)


