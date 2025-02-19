#Funktion til at indlæse csv-filen som en string
def læs_fil(fil):
    with open(fil, 'r') as file:
        text = file.read().strip()
    return text

tekst = læs_fil("navn på fil")