import requests
import tensorflow as tf
import numpy as np
import json
 
# Asetukset avoimelle säätietopalvelulle
OPENWEATHERMAP_API_KEY = 'f42bcf82d75f27fbcb91d57e7adf4e92'
CITIES = ['Helsinki', 'Turku', 'Tampere', 'Oulu']
OPENWEATHERMAP_URL = f'http://api.openweathermap.org/data/2.5/weather?q={{city}}&appid={OPENWEATHERMAP_API_KEY}'
 
def hae_saaennuste_openweathermap(api_key, cities):
    saaennusteet = {}
    for city in cities:
        try:
            # Haetaan säätiedot OpenWeatherMap -palvelusta
            url = OPENWEATHERMAP_URL.format(city=city)
            vastaus = requests.get(url)
            vastaus.raise_for_status()  # Tarkista HTTP-virheet
            tiedot = vastaus.json()
            # Lisää tarkistus varmistaaksesi, että tiedot ovat olemassa
            if 'main' in tiedot and 'temp' in tiedot['main'] and 'wind' in tiedot and 'speed' in tiedot['wind']:
                saaennusteet[city] = {
                    'lämpötila': tiedot['main']['temp'],
                    'tuulen_nopeus': tiedot['wind']['speed'],
                }
            else:
                print(f"Virhe kaupungin {city} säätietojen hakemisessa: Puuttuvat tiedot")
        except Exception as e:
            print(f"Virhe kaupungin {city} säätietojen hakemisessa: {e}")
    return saaennusteet
 
def ennusta_sahkon_hinta(saaennusteet):
    # Käytä tensorflow-kirjastoa ennustamiseen
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='relu')  # Vaihdan aktivoinnin reluksi
])
 
 
    # Haetaan säätiedot OpenWeatherMap -palvelusta
    saaennusteet_openweathermap = hae_saaennuste_openweathermap(OPENWEATHERMAP_API_KEY, CITIES)
 
    # Muokkaa syötteet ja tulosteet numpy-taulukoiksi
    x_train = np.array([[saaennusteet_openweathermap[city]['lämpötila'], saaennusteet_openweathermap[city]['tuulen_nopeus']] for city in CITIES], dtype=np.float32)
 
    # Määrittää odotetut tehot (fiktiiviset sähkön hinnat)
    y_train = np.array([
    [25.0],  # Helsinki
    [23.0],  # Turku
    [28.0],  # Tampere
    [20.0],  # Oulu
], dtype=np.float32)
 
    # Harjoittele modelin
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100)
 
    # Modelin ennustus
    for city in saaennusteet:
        input_data = np.array([saaennusteet[city]['lämpötila'], saaennusteet[city]['tuulen_nopeus']], dtype=np.float32).reshape(1, 2)
        ennuste = model.predict(input_data)
        print(f"Sääennuste kaupungille {city}: {saaennusteet_openweathermap[city]}")
        print(f"Sähkön hinta ennuste kaupungille {city}: {ennuste[0][0]}")
 
if __name__ == "__main__":
    # Haetaan säätiedot OpenWeatherMap -palvelusta
    saaennusteet_openweathermap = hae_saaennuste_openweathermap(OPENWEATHERMAP_API_KEY, CITIES)
    print("Sääennusteet OpenWeatherMap -palvelusta:")
    print(json.dumps(saaennusteet_openweathermap, indent=2))
 
    # Ennustetaan sähkön hinta
    print("\nEnnustetaan sähkön hinta:")
    ennusta_sahkon_hinta(saaennusteet_openweathermap)
