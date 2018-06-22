import argparse
import dataset
import os

from torch.utils.data import DataLoader
from torchvision import utils

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--letter', type=str, default='S', help='the letter to seperate') # defaults to 's' as this letter is most often present in the dataset
parser.add_argument('--feature', type=str, default='countries', help='the feature for which to extract the data')
opt = parser.parse_args()
print(opt)

# Some input error checking and dataformatting
if len(opt.letter) > 1 or not opt.letter.isalpha():
	raise ValueError('Input can only be one letter')
else:
	print('Will seperate letter \'' + opt.letter + '\'')
sep_letter = opt.letter.upper()

# directories to save the output
base_directory = 'custom_datasets'
directory = base_directory + '/Seperated_' + opt.feature + '_' + sep_letter
if not os.path.exists(directory):
    os.makedirs(directory)

# This opens the HDF5 database and allows us to query it
data = dataset.get_dataset()

# Make seperate counter for the images
image_id = 0

# Add header row to the csv file and determind the features
if opt.feature == 'countries':
	header_row = 'File_name,FR,CH,DE,ES,IT,BE,GB,POR,NL,DK,CZ'
	final_features = ['FR', 'CH', 'DE', 'ES', 'IT', 'BE', 'GB', 'POR', 'NL', 'DK', 'CZ']
elif opt.feature == 'cities':
	header_row = 'File_name,Lyon,Paris,Basel,Zwickau,Valencia,Frankfurt,Brescia,Madrid,Antwerpen,Venezia,Firenze,Oppenheim,Munchen,Magdenburg,Strasbourg,London,Sevilla,Koln,Louvain,Zurich,Worms,Bologna,Ingoldstadt,Ferrara,Marburg,Geneve,Pavia,Bergamo,Neustadt,Mainz,Heidelberg,Hagenau,Leipzig,Jena,Augsburg,Wesel,Macerata,Como,Nuremberg,Douai,Vincenza,Alcala,Wittenberg,Dortmund,Morges,Lucca,Lisboa,Dillingen,Rome,Alkmaar,Amsterdam,Hamburg,Leiden,Ieper,Kopenhagen,Freiburg,Zaragozza,Troyes,Salamanca,Erfurt,Poitiers,Wurzburg,Genova,Caen,Praha'
	final_features = ['Lyon', 'Paris', 'Basel', 'Zwickau', 'Valencia', 'Frankfurt', 'Brescia', 'Madrid', 'Antwerpen', 'Venezia', 'Firenze', 'Oppenheim', 'Munchen', 'Magdenburg', 'Strasbourg', 'London', 'Sevilla', 'Koln', 'Louvain', 'Zurich', 'Worms', 'Bologna', 'Ingoldstadt', 'Ferrara', 'Marburg', 'Geneve', 'Pavia', 'Bergamo', 'Neustadt', 'Mainz', 'Heidelberg', 'Hagenau', 'Leipzig', 'Jena', 'Augsburg', 'Wesel', 'Macerata', 'Como', 'Nuremberg', 'Douai', 'Vincenza', 'Alcala', 'Wittenberg', 'Dortmund', 'Morges', 'Lucca', 'Lisboa', 'Dillingen', 'Rome', 'Alkmaar', 'Amsterdam', 'Hamburg', 'Leiden', 'Ieper', 'Kopenhagen', 'Freiburg', 'Zaragozza', 'Troyes', 'Salamanca', 'Erfurt', 'Poitiers', 'Wurzburg', 'Genova', 'Caen', 'Praha']
elif opt.feature == 'names':
	header_row = 'File_name,Simon_Bevilaqua,Matthieu_David,Valentin_Curio,Gabriel_Kantz,Diego_Gumiel,Jean_Pillehotte,Andreas_Cratander,Sebastien_Cramoisy,Seitz_Heirs_Peter_Main,Andre_Bocard,Comino_Presegni,Pedro_Madrigal,Oudin_Petit,Willem_Vorsterman,Gaspare_2_Bindoni,Matthias_David,Pre_Galliot_Du,Philippo_2_Giunta,Jakob_Kobel,Adam_Berg,Philippo_1_Giunta,Antoine_Vincent,Wolfgang_Kirchener,Gabriel_Buon,Johann_Knobloch,Guillaume_Cavellat,Gilles_Gourbin,Vincenzo_Sabbio,Becker_Matthias_Main,Robert_Barker,Bonetus_Locatellus,Jacob_Cromberger,Bartholomaeus_Vincent,Mylius_Crato,Antoine_Augerelle,Robert_Chaudiere,Hero_Fuchs,Mace_Bonhomme,Lucantonio_1_Giunta,Charlotte_Guillard,Lucantonio_2_Giunta,Claude_Servain,Peter_Quentell,Jacques_Huguetan,Arnold_Heirs_Birckman,Dirck_Martensz,Giovanni_Battista_Ciotti,Bernardino_Gerrualda,Claude_Chevallon,Andreas_Gessner,Simon_Beys,Peter_Schoeffer,Johann_Oporinus,Heinrich_Petri,Sebastien_Honorat,Giovanni_Rosso,Girolamo_Scoto,Robert_Winter,Bartollomeo_Carampello,Alexander_Weissenhorn,Vittorio_Baldini,Candido_Benedetto,Simon_Colines,Adam_Petri,Jean_Roigny,Jacques_Sacon,Christian_Egenolff,Sebastian_Henricpetri,Giovanni_Maria_Bonelli,Johann_Soter,Eustache_Vignon,Giacomo_Pocatello,Antoine_Ry,Ludwig_Ostenius,Jamet_Mettayer,Jean_Louis,Jean_Tournes,Feyerabend_Johann_Main,Joannes_Moylin,Etienne_Gueynard,Gerwin_Calenius,Francesco_Rampazetto,Comino_Ventura,Johann_Schott,Pierre_Marechal,Wilhelm_Harnisch,Claude_Fremy,Sebastien_Griffo,Gregorio_Gregoriis,Josse_Bade,Lorenzo_Torrentino,Francesco_Franceschi,Franz_Behem,Melchior_Neuss,Francesco_Rossi,Cyriacus_Jacob_Main,Andreas_Cambier,Domenico_Farri,Ivo_Schoeffer,Heinrich_Gran,Michael_Blum,Heirs_Thomas_Rebart,Peter_Perna,Melchior_Lotter,Ludwig_Alectorius,David_Sartorius,Wechel_Heirs_Andreas_Main,Paganinus_Paganinis,Ernhard_Ratdolt,Johann_Walder,Heinrich_Stayner,Porte_Hugues_La,Hans_Braker,Johann_Miller,G_M_Beringen,Johann_Feyerabend,Thielman_Kerver,Ulrich_Gering,Arnold_Birckman,Vincent_Portonariis,Robert_Estienne,Aldus_1_Manutius,Jaques_Marechal,Braubach_Peter_Main,Nicolas_Benedictis,Valentin_Kobian,Sebastiano_Martellini,Prez_Nicolas_Des,Jerome_Olivier,Francois_Regnault,Michael_Sonnius,Bartolommeo_Zanis,Pierre_Fradin,Johann_Herwagen,Egenolff_Christian_Main,Arnoul_Angelier,Laurent_Sonnius,Claude_Senneton,Thomas_Anshelm,Theodore_Rihel,Andre_Wechel,Guillaume_Chaudiere,Symphorien_Barbier,Melchior_Sessa,Joannes_Crispinus,Christian_Wechel,Johan_Bebel,Antoine_Harsy,Eucharius_Cervicorius,Lechler_Martin_Main,Anselmo_Giacarelli,Jean_le_Preux,Michael_Hillenius,Porte_Sybille_La,Gottardo_da_Ponte,Berthold_Rembold,Johann_von_Berg,Blaise_Guido,GB_Bellagamba,Willem_Sylvius,Georg_Hantsch,Damiano_Zenaro,Michel_Vascosan,Jan_Loe,Adrien_Tunebre,Giovanni_Battista_Phaelli,Conrad_Bade,Jeune_Martin_Le,Allesandro_Benacci,Jan_1_Roelants,Jean_Bogard,Nicolaus_Episcopius,Vincent_Valgrisi,George_Bishop,Gilbert_Villiers,Johann_Quentell,Theobaldus_Ancelin,Richter_Wolfgang_Main,Noir_Guillaume_Le,Michael_Isengrin,Guillaume_Rouille,Jean_Clein,Joannes_Platea,Pierre_Vidoue,Sebastien_Nivelle,Louis_Cynaeus,Jean_Le_Rouge,Christophorus_Zelle,Giorgio_Greco,Arnaldo_Guillen_Brocar,Hieronymus_Verdussen,Laurentius_Annison,Jakob_Kundig,Guillaume_Guillemot,Christoph_Rodinger,Symon_Cock,Claude_Davost,Charles_Estienne,Jan_1_Moretus,Johann_Schwertel,Francesco_Bolzetta,Fevre_Francois_Le,Melchior_Soter,Denys_Janot,Johann_Schoeffer,Guillaume_Lairmarius,Venturina_Rosselini,Eichorn_Andreas_Oder,Jacques_Faure,Barthelemy_Ancelin,Vincenzo_Busdraghi,Martin_LEmpereur,Johann_Neuss,Hartmann_Friedrich_Oder,Pierre_Galterus,Hieronymus_Wallaeus,Rutgerus_Velpius,Levinus_Hulsius,de_Ferrara_Gabriele_Giolito,Antonio_Padovani,Jean_Cavelleir,Bassaeus_Oder,Johann_Faber,Ambrosius_Froben,Guichard_Julieron,Hieronymus_Commelin,Antoine_Blanchard,Heirs_Symphorien_Beraud,Jacques_Myt,Godefriedus_Kempen,Puys_Jacques_Du,Andrea_Arrivabene,Heinrich_Quentell,Froschauer_Christopher_1_CH,Miguel_Eguia,Pierre_Roussin,Caspar_Behem,Luis_Rodrigues,Henry_Estienne,Samuel_Konig,Guglielmo_Fontaneto,Giovanni_Guerigli,Mathieu_Berjon,Anton_Hierat,Sebald_Mayer,Johann_Prael,Widow_Gabriel_Buon,Heinrich_Gymnicus,Thomas_Courteau,Symphorien_Beraud,Bonaventura_Nugo,Joannes_Criegher,Jan_van_Keerbergen,Bartolomeo_Bonfadini,Petrus_Colinaeus,Heirs_Sebastien_Griffo,Francois_Fradin,Guillaume_Morel,Hieronymus_Froben,Widow_Martinus_Nutius,Johann_Froben,Nicolaus_Brylinger,Vivant_Gautherot,Widow_Hendrick_Peetersen,Joachim_Trognesius,Nicolaus_Faber,Jean_Petit,Thibaud_Payen,Johann_Setzer,Jean_Crespin,Norton_Johann_Main,Jacques_Androuet,Giovanni_Bariletto,Hans_Luft,Jean_Ogerolles,Andrea_Poleti,Jacob_Meester,Christoffel_Cunradus,van_Waesberghe_Joannes_Janssonius,Maternus_Cholinus,Thomas_Raynald,Johann_Ruremundensis,Jean_Laon,Giorgio_Angelieri,Andreas_Angermaier,Jean_Gerard,Martinus_Verhasselt,Federic_Morel,Elisabetta_Rusconi,Georg_Papen,Jean_Dalier,Jean_Bienne,Heirs_Hieronymus_Benedictis,Barezzo_Barezzi,Thomas_Wolf,Martinus_Gymnicus,Etienne_Dolet,Francesco_Suzzi,Froschauer_Christopher_2_CH,Clerc_David_Le,Konrad_Mechel,Christoffel_Guyot,Arnout_Brakel,Marcus_Zaltieri,Joannes_Degaram,Joannes_Masius,Henri_Estienne,Horace_Cardon,Hadrianus_Perier,Joos_Destree,Peter_Seitz,Benoit_Prevost,Johann_Gruninger,Mats_Vingaard,Konrad_Caesarius,Jean_Marion,Michel_Cotier,Jacob_Roussin,Breisgau_Emmeus_Johann_im,Pierre_Mettayer,Francisco_Baba,Guillaume_Julianus,Johann_Birckman,Catharina_Gerlach,Pedro_Bernuz,Rouge_Nicolas_Le,Jacob_Stoer,Guillaume_Foquel,Konrad_Waldkirch,Paul_Frellon,Fredericus_Lydius,Bartollomeo_Alberti,Pasqier_Le_Tellier,Joannes_Grapheus,Guillaume_Lairmarie,Paulus_Queckus,Hendrik_Connix,Raben_Georg_Main,Johann_Gymnicus,Pietro_Maria_Marchetti,Wolfgang_Sthurmer,Philippe_Tinghi,Pamphilius_Gengenbach,Johann_Schoenstenius,Jasper_Gennep,Herman_Moller,Pierre_Gautier,Bernardino_Vitalis,Jean_Blanchet,Thomas_Brumennius,Guillaume_Merlin,Bartholomaeus_Westheimer,Elzeviers,Heinrich_von_Aich,Georg_Defner,Antoine_Chuppin,Johann_Gemusaeus,Antonio_Bellona,Girard_Angier,Giovanni_1_Griffio,Ottavio_Scoto,Francois_Arnoullet,Heirs_Johann_Quentell,Robert_Field,Andreas2_en_HJ_Gesner,Wilhelm_Lutzenkirchen,Bolognino_Zaltieri,Georg_Schwartz,Christoffel_Plantin,Gilles_Huguetan,Baldassare_Constantini,Gerard_Morrhe,Jacques_Giunta,Bocchiana_Nova_Academia'
	final_features = ['Simon Bevilaqua', 'Matthieu David', 'Valentin Curio', 'Gabriel Kantz', 'Diego Gumiel', 'Jean Pillehotte', 'Andreas Cratander', 'Sebastien Cramoisy', 'Seitz Heirs Peter Main', 'Andre Bocard', 'Comino Presegni', 'Pedro Madrigal', 'Oudin Petit', 'Willem Vorsterman', 'Gaspare 2 Bindoni', 'Matthias David', 'Pre Galliot Du', 'Philippo 2 Giunta', 'Jakob Kobel', 'Adam Berg', 'Philippo 1 Giunta', 'Antoine Vincent', 'Wolfgang Kirchener', 'Gabriel Buon', 'Johann Knobloch', 'Guillaume Cavellat', 'Gilles Gourbin', 'Vincenzo Sabbio', 'Becker Matthias Main', 'Robert Barker', 'Bonetus Locatellus', 'Jacob Cromberger', 'Bartholomaeus Vincent', 'Mylius Crato', 'Antoine Augerelle', 'Robert Chaudiere', 'Hero Fuchs', 'Mace Bonhomme', 'Lucantonio 1 Giunta', 'Charlotte Guillard', 'Lucantonio 2 Giunta', 'Claude Servain', 'Peter Quentell', 'Jacques Huguetan', 'Arnold Heirs Birckman', 'Dirck Martensz', 'Giovanni Battista Ciotti', 'Bernardino Gerrualda', 'Claude Chevallon', 'Andreas Gessner', 'Simon Beys', 'Peter Schoeffer', 'Johann Oporinus', 'Heinrich Petri', 'Sebastien Honorat', 'Giovanni Rosso', 'Girolamo Scoto', 'Robert Winter', 'Bartollomeo Carampello', 'Alexander Weissenhorn', 'Vittorio Baldini', 'Candido Benedetto', 'Simon Colines', 'Adam Petri', 'Jean Roigny', 'Jacques Sacon', 'Christian Egenolff', 'Sebastian Henricpetri', 'Giovanni Maria Bonelli', 'Johann Soter', 'Eustache Vignon', 'Giacomo Pocatello', 'Antoine Ry', 'Ludwig Ostenius', 'Jamet Mettayer', 'Jean Louis', 'Jean Tournes', 'Feyerabend Johann Main', 'Joannes Moylin', 'Etienne Gueynard', 'Gerwin Calenius', 'Francesco Rampazetto', 'Comino Ventura', 'Johann Schott', 'Pierre Marechal', 'Wilhelm Harnisch', 'Claude Fremy', 'Sebastien Griffo', 'Gregorio Gregoriis', 'Josse Bade', 'Lorenzo Torrentino', 'Francesco Franceschi', 'Franz Behem', 'Melchior Neuss', 'Francesco Rossi', 'Cyriacus Jacob Main', 'Andreas Cambier', 'Domenico Farri', 'Ivo Schoeffer', 'Heinrich Gran', 'Michael Blum', 'Heirs Thomas Rebart', 'Peter Perna', 'Melchior Lotter', 'Ludwig Alectorius', 'David Sartorius', 'Wechel Heirs Andreas Main', 'Paganinus Paganinis', 'Ernhard Ratdolt', 'Johann Walder', 'Heinrich Stayner', 'Porte Hugues La', 'Hans Braker', 'Johann Miller', 'G M Beringen', 'Johann Feyerabend', 'Thielman Kerver', 'Ulrich Gering', 'Arnold Birckman', 'Vincent Portonariis', 'Robert Estienne', 'Aldus 1 Manutius', 'Jaques Marechal', 'Braubach Peter Main', 'Nicolas Benedictis', 'Valentin Kobian', 'Sebastiano Martellini', 'Prez Nicolas Des', 'Jerome Olivier', 'Francois Regnault', 'Michael Sonnius', 'Bartolommeo Zanis', 'Pierre Fradin', 'Johann Herwagen', 'Egenolff Christian Main', 'Arnoul Angelier', 'Laurent Sonnius', 'Claude Senneton', 'Thomas Anshelm', 'Theodore Rihel', 'Andre Wechel', 'Guillaume Chaudiere', 'Symphorien Barbier', 'Melchior Sessa', 'Joannes Crispinus', 'Christian Wechel', 'Johan Bebel', 'Antoine Harsy', 'Eucharius Cervicorius', 'Lechler Martin Main', 'Anselmo Giacarelli', 'Jean le Preux', 'Michael Hillenius', 'Porte Sybille La', 'Gottardo da Ponte', 'Berthold Rembold', 'Johann von Berg', 'Blaise Guido', 'GB Bellagamba', 'Willem Sylvius', 'Georg Hantsch', 'Damiano Zenaro', 'Michel Vascosan', 'Jan Loe', 'Adrien Tunebre', 'Giovanni Battista Phaelli', 'Conrad Bade', 'Jeune Martin Le', 'Allesandro Benacci', 'Jan 1 Roelants', 'Jean Bogard', 'Nicolaus Episcopius', 'Vincent Valgrisi', 'George Bishop', 'Gilbert Villiers', 'Johann Quentell', 'Theobaldus Ancelin', 'Richter Wolfgang Main', 'Noir Guillaume Le', 'Michael Isengrin', 'Guillaume Rouille', 'Jean Clein', 'Joannes Platea', 'Pierre Vidoue', 'Sebastien Nivelle', 'Louis Cynaeus', 'Jean Le Rouge', 'Christophorus Zelle', 'Giorgio Greco', 'Arnaldo Guillen Brocar', 'Hieronymus Verdussen', 'Laurentius Annison', 'Jakob Kundig', 'Guillaume Guillemot', 'Christoph Rodinger', 'Symon Cock', 'Claude Davost', 'Charles Estienne', 'Jan 1 Moretus', 'Johann Schwertel', 'Francesco Bolzetta', 'Fevre Francois Le', 'Melchior Soter', 'Denys Janot', 'Johann Schoeffer', 'Guillaume Lairmarius', 'Venturina Rosselini', 'Eichorn Andreas Oder', 'Jacques Faure', 'Barthelemy Ancelin', 'Vincenzo Busdraghi', 'Martin LEmpereur', 'Johann Neuss', 'Hartmann Friedrich Oder', 'Pierre Galterus', 'Hieronymus Wallaeus', 'Rutgerus Velpius', 'Levinus Hulsius', 'de Ferrara Gabriele Giolito', 'Antonio Padovani', 'Jean Cavelleir', 'Bassaeus Oder', 'Johann Faber', 'Ambrosius Froben', 'Guichard Julieron', 'Hieronymus Commelin', 'Antoine Blanchard', 'Heirs Symphorien Beraud', 'Jacques Myt', 'Godefriedus Kempen', 'Puys Jacques Du', 'Andrea Arrivabene', 'Heinrich Quentell', 'Froschauer Christopher 1 CH', 'Miguel Eguia', 'Pierre Roussin', 'Caspar Behem', 'Luis Rodrigues', 'Henry Estienne', 'Samuel Konig', 'Guglielmo Fontaneto', 'Giovanni Guerigli', 'Mathieu Berjon', 'Anton Hierat', 'Sebald Mayer', 'Johann Prael', 'Widow Gabriel Buon', 'Heinrich Gymnicus', 'Thomas Courteau', 'Symphorien Beraud', 'Bonaventura Nugo', 'Joannes Criegher', 'Jan van Keerbergen', 'Bartolomeo Bonfadini', 'Petrus Colinaeus', 'Heirs Sebastien Griffo', 'Francois Fradin', 'Guillaume Morel', 'Hieronymus Froben', 'Widow Martinus Nutius', 'Johann Froben', 'Nicolaus Brylinger', 'Vivant Gautherot', 'Widow Hendrick Peetersen', 'Joachim Trognesius', 'Nicolaus Faber', 'Jean Petit', 'Thibaud Payen', 'Johann Setzer', 'Jean Crespin', 'Norton Johann Main', 'Jacques Androuet', 'Giovanni Bariletto', 'Hans Luft', 'Jean Ogerolles', 'Andrea Poleti', 'Jacob Meester', 'Christoffel Cunradus', 'van Waesberghe Joannes Janssonius', 'Maternus Cholinus', 'Thomas Raynald', 'Johann Ruremundensis', 'Jean Laon', 'Giorgio Angelieri', 'Andreas Angermaier', 'Jean Gerard', 'Martinus Verhasselt', 'Federic Morel', 'Elisabetta Rusconi', 'Georg Papen', 'Jean Dalier', 'Jean Bienne', 'Heirs Hieronymus Benedictis', 'Barezzo Barezzi', 'Thomas Wolf', 'Martinus Gymnicus', 'Etienne Dolet', 'Francesco Suzzi', 'Froschauer Christopher 2 CH', 'Clerc David Le', 'Konrad Mechel', 'Christoffel Guyot', 'Arnout Brakel', 'Marcus Zaltieri', 'Joannes Degaram', 'Joannes Masius', 'Henri Estienne', 'Horace Cardon', 'Hadrianus Perier', 'Joos Destree', 'Peter Seitz', 'Benoit Prevost', 'Johann Gruninger', 'Mats Vingaard', 'Konrad Caesarius', 'Jean Marion', 'Michel Cotier', 'Jacob Roussin', 'Breisgau Emmeus Johann im', 'Pierre Mettayer', 'Francisco Baba', 'Guillaume Julianus', 'Johann Birckman', 'Catharina Gerlach', 'Pedro Bernuz', 'Rouge Nicolas Le', 'Jacob Stoer', 'Guillaume Foquel', 'Konrad Waldkirch', 'Paul Frellon', 'Fredericus Lydius', 'Bartollomeo Alberti', 'Pasqier Le Tellier', 'Joannes Grapheus', 'Guillaume Lairmarie', 'Paulus Queckus', 'Hendrik Connix', 'Raben Georg Main', 'Johann Gymnicus', 'Pietro Maria Marchetti', 'Wolfgang Sthurmer', 'Philippe Tinghi', 'Pamphilius Gengenbach', 'Johann Schoenstenius', 'Jasper Gennep', 'Herman Moller', 'Pierre Gautier', 'Bernardino Vitalis', 'Jean Blanchet', 'Thomas Brumennius', 'Guillaume Merlin', 'Bartholomaeus Westheimer', 'Elzeviers', 'Heinrich von Aich', 'Georg Defner', 'Antoine Chuppin', 'Johann Gemusaeus', 'Antonio Bellona', 'Girard Angier', 'Giovanni 1 Griffio', 'Ottavio Scoto', 'Francois Arnoullet', 'Heirs Johann Quentell', 'Robert Field', 'Andreas2 en HJ Gesner', 'Wilhelm Lutzenkirchen', 'Bolognino Zaltieri', 'Georg Schwartz', 'Christoffel Plantin', 'Gilles Huguetan', 'Baldassare Constantini', 'Gerard Morrhe', 'Jacques Giunta', 'Bocchiana Nova Academia']
else:
	raise ValueError('This is a not a feature to seperate on: ' + opt.feature)

with open(base_directory + '/list_attr_' + opt.feature + '_' + sep_letter + '.csv', 'a') as myfile:
	myfile.write(header_row)


for i in range(len(data)):

	# Extract the data from the HDF5 file
	image, letter, country, city, name = data.__getitem__(i)

	# Only save the image and metadata if it matches the seperated letter
	if letter == sep_letter:
		# Format image names
		nr = len(str(image_id))
		if nr == 1:
			image_name = '0000' + str(image_id)
		elif nr == 2:
			image_name = '000' + str(image_id)
		elif nr == 3:
			image_name= '00' + str(image_id)
		elif nr == 4:
			image_name = '0' + str(image_id)
		else: 
			image_name = str(image_id)

		# Save the image
		utils.save_image(image, directory + '/' + image_name + '.jpg')
		
		# Create a new row for the file
		newrow = image_name + '.jpg'

		# Find the features that apply to the image
		for feature in final_features:
			# 1 for present features, 0 for all other
			if feature == letter or feature == country or feature == city or feature == name:
				newrow += ',1'
			else:
				newrow += ',0'

		# Write the row to the file
		with open(base_directory + '/list_attr_' + opt.feature + '_' + sep_letter + '.csv', 'a') as myfile:
			myfile.write("\n" + newrow)

		# Increase image counter	
		image_id += 1