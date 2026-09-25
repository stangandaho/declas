"""English / French interface text.

English is written directly in the code. tr() swaps it for French when that
language is selected in Setting > General (applied at the next start).
Strings with values use {placeholders}: tr("Entry {n}").format(n=3).
"""

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (QWidget, QAction, QAbstractButton, QLabel, QLineEdit,
                             QTextEdit, QPlainTextEdit, QGroupBox, QTabWidget, QMenu,
                             QTableWidget)

LANGUAGE = QSettings("Declas", "Declas").value("language", "en", type=str)

FRENCH = {
    # Main window (layout file)
    "Metadata": "Métadonnées",
    "Map": "Carte",
    "Inference": "Inférence",
    "Apply": "Appliquer",
    "View": "Afficher",
    "Edit": "Modifier",
    "Setting": "Paramètres",
    "Appearance": "Apparence",
    "Models": "Modèles",
    "File": "Fichier",
    "Tool Bar": "Barre d'outils",
    "Inference parameters": "Paramètres d'inférence",
    "Language": "Langue",
    "Import": "Importer",
    "Quit": "Quitter",
    "Build table": "Construire le tableau",
    "Dark": "Sombre",
    "Light": "Clair",
    "System": "Système",

    # Toolbar and menus
    "Select directory": "Sélectionner un dossier",
    "Select media": "Sélectionner un média",
    "Select a single image or video": "Sélectionner une image ou une vidéo",
    "Zoom lens": "Loupe",
    "Hover over the image to magnify": "Survolez l'image pour l'agrandir",
    "Draw bounding box": "Dessiner une boîte englobante",
    "Drag on the image to add a box around a missed animal":
        "Faites glisser sur l'image pour encadrer un animal manqué",
    "Delete bounding box": "Supprimer une boîte englobante",
    "Hover a box and click it to delete it": "Survolez une boîte puis cliquez pour la supprimer",
    "Show on map": "Afficher sur la carte",
    "Run on current media": "Exécuter sur le média affiché",
    "Run folder(s)": "Exécuter sur le(s) dossier(s)",
    "Build table from detection/classification": "Construire le tableau des détections/classifications",
    "Clear all detection/classification reports": "Effacer tous les rapports de détection/classification",
    "Target or No target split": "Séparer les médias avec / sans cible",
    "Extensions": "Extensions",
    "Browse and install model extensions from the online registry":
        "Parcourir et installer des extensions de modèles depuis le registre en ligne",
    "Publish": "Publier",
    "Learn how to publish your own model extension":
        "Découvrir comment publier votre propre extension de modèle",
    "Tags": "Étiquettes",
    "Define custom tags": "Définir des étiquettes personnalisées",
    "General": "Général",

    # Inference and tags panels
    "Add species": "Ajouter une espèce",
    "Species": "Espèce",
    "Species name": "Nom de l'espèce",
    "Count": "Nombre",
    "Remove this entry": "Supprimer cette entrée",
    "Custom Tags": "Étiquettes personnalisées",
    " Add entry ": " Ajouter une entrée ",
    "No tags defined. Go to Setting > Tags to add some.":
        "Aucune étiquette définie. Allez dans Paramètres > Étiquettes pour en ajouter.",
    "Entry {n}": "Entrée {n}",
    "Tags saved": "Étiquettes enregistrées",

    # General settings
    "General Settings": "Paramètres généraux",
    "Notification": "Notification",
    "Enable notification sound on completion": "Jouer un son à la fin d'un traitement",
    "Sound:": "Son :",
    "Language:": "Langue :",
    "Theme:": "Thème :",
    "Cancel": "Annuler",
    "OK": "OK",

    # Files, runs and reports
    "Select json file": "Sélectionner un fichier JSON",
    "File imported ✅": "Fichier importé ✅",
    "Select Folder": "Sélectionner un dossier",
    "All media": "Tous les médias",
    "Images": "Images",
    "Videos": "Vidéos",
    "Error: {error}": "Erreur : {error}",
    "Split applied ✅": "Séparation effectuée ✅",
    "Video processing is disabled. Enable it in Inference Parameters.":
        "Le traitement des vidéos est désactivé. Activez-le dans les paramètres d'inférence.",
    "Running…": "Exécution…",
    "Extension '{name}' weights not downloaded.":
        "Les poids de l'extension « {name} » ne sont pas téléchargés.",
    "Done ✅": "Terminé ✅",
    "Change applied ✅": "Modification appliquée ✅",
    "Clear reports": "Effacer les rapports",
    "Remove all detections.json and *.csv files under:\n{folder}\n\nThis cannot be undone.":
        "Supprimer tous les fichiers detections.json et *.csv dans :\n{folder}\n\n"
        "Cette action est irréversible.",
    "Cleared {count} report file(s) ✅": "{count} fichier(s) de rapport supprimé(s) ✅",
    "Table built successfully and saved at {path}": "Tableau construit et enregistré dans {path}",
    "Extensions reloaded.": "Extensions rechargées.",
    "Completed successfully": "Terminé avec succès",
    "Detections found – running distance estimation…":
        "Détections trouvées : lancement de l'estimation de distance…",
    "No media files found in directory.": "Aucun média trouvé dans le dossier.",
    "Extension weights missing.": "Poids de l'extension manquants.",
    "No media found or error occurred": "Aucun média trouvé ou une erreur est survenue",
    "❌ No valid subdirectories found. Select the parent folder that contains station sub-folders.":
        "❌ Aucun sous-dossier valide trouvé. Sélectionnez le dossier parent "
        "qui contient les sous-dossiers des stations.",
    "❌ No processable folders found. Check your directory structure.":
        "❌ Aucun dossier exploitable trouvé. Vérifiez la structure de vos dossiers.",
    "Unexpected error: {error}": "Erreur inattendue : {error}",
    "🔄 Processing: {name}": "🔄 Traitement : {name}",
    "🔄 Station: {name}": "🔄 Station : {name}",
    "⚠️  Skipping '{name}': no media found": "⚠️  « {name} » ignoré : aucun média trouvé",
    "❌ Error in {station}: {error}": "❌ Erreur dans {station} : {error}",
    "Processing video: {name}": "Traitement de la vidéo : {name}",
    "No frames extracted.": "Aucune image extraite.",
    "Running inference on {count} frames…": "Inférence sur {count} images…",
    "🎉 Video processed.": "🎉 Vidéo traitée.",

    # Bounding box editing
    "This detection file uses an old format and cannot be edited.":
        "Ce fichier de détection utilise un ancien format et ne peut pas être modifié.",
    "New bounding box": "Nouvelle boîte englobante",
    "Species:": "Espèce :",
    "Box added: {species}": "Boîte ajoutée : {species}",
    "Box deleted: {species}": "Boîte supprimée : {species}",

    # Distance estimation
    "No FOV configured": "Aucun champ de vision configuré",
    "No Field-Of-View data is set for any station.\n\n"
    "Distance will be estimated as raw line-of-sight depth (no angular correction).\n\n"
    "To improve accuracy, open Inference Parameters → Set Field Of View per Station.":
        "Aucun champ de vision n'est défini pour les stations.\n\n"
        "La distance sera estimée à partir de la profondeur brute (sans correction angulaire).\n\n"
        "Pour plus de précision, ouvrez Paramètres d'inférence → Définir le champ de vision.",
    "Distance estimation failed": "L'estimation de distance a échoué",
    "Depth model '{name}' not found. Install it via the Extension Manager.":
        "Modèle de profondeur « {name} » introuvable. Installez-le depuis le gestionnaire d'extensions.",
    "Distance estimation: loading depth model…": "Estimation de distance : chargement du modèle de profondeur…",
    "Distance estimation: running…": "Estimation de distance : en cours…",
    "No detections.json found for distance estimation.":
        "Aucun fichier detections.json trouvé pour l'estimation de distance.",
    "⚠️ No FOV match for station '{station}' — raw depth used.":
        "⚠️ Aucun champ de vision pour la station « {station} » : profondeur brute utilisée.",
    "Distance estimation: {pct}% for {station}": "Estimation de distance : {pct} % pour {station}",
    "Distance estimation complete ✅": "Estimation de distance terminée ✅",
    "Distance estimation error: {error}": "Erreur d'estimation de distance : {error}",
    "Downloading model: {done:.0f}/{total:.0f} MB  {pct}%":
        "Téléchargement du modèle : {done:.0f}/{total:.0f} Mo  {pct} %",
    "Downloading depth model (first time only)…":
        "Téléchargement du modèle de profondeur (première fois uniquement)…",

    # Message boxes
    "Error": "Erreur",
    "Choose a directory that contains images": "Choisissez un dossier contenant des images",
    "Missed directory": "Dossier manquant",
    "Choose an image, directory that contains images, or import a detection/classification file.":
        "Choisissez une image, un dossier contenant des images, "
        "ou importez un fichier de détection/classification.",
    "Missed path": "Chemin manquant",
    "Successful": "Succès",
    "No GPS info found": "Aucune information GPS trouvée",
    "Empty metadata": "Métadonnées vides",
    "No model installed. Add one!": "Aucun modèle installé. Ajoutez-en un !",
    "No model": "Aucun modèle",
    "The edit is invalid. Check and try again": "La modification n'est pas valide. Vérifiez et réessayez.",
    "Invalid edit": "Modification non valide",

    # Inference parameters
    "Detection": "Détection",
    "Classification": "Classification",
    "Image size:": "Taille d'image :",
    "Confidence threshold:": "Seuil de confiance :",
    "Maximum detection:": "Détections maximum :",
    "Classes of interest:": "Classes d'intérêt :",
    "Process video": "Traiter les vidéos",
    "When checked, video files will be processed alongside images.":
        "Si coché, les vidéos seront traitées en même temps que les images.",
    "Frame stride:": "Intervalle d'images :",
    "Half-precision": "Demi-précision",
    "Run on main directory": "Exécuter sur le dossier principal",
    "Device:": "Périphérique :",
    "Task:": "Tâche :",
    "Model:": "Modèle :",
    "Distance estimation": "Estimation de distance",
    "Estimate": "Estimer",
    "Depth model:": "Modèle de profondeur :",
    "Set Field Of View": "Définir le champ de vision",
    "No models installed": "Aucun modèle installé",
    "No depth models installed": "Aucun modèle de profondeur installé",
    "Field Of View per Station": "Champ de vision par station",
    "Station": "Station",
    "FOV (degrees)": "Champ de vision (degrés)",
    "Add": "Ajouter",
    "Remove": "Supprimer",
    "Upload CSV": "Importer un CSV",
    "CSV format with one row per station.": "Format CSV avec une ligne par station.",
    "Open CSV": "Ouvrir un CSV",
    "CSV files (*.csv *.txt)": "Fichiers CSV (*.csv *.txt)",
    "CSV imported": "CSV importé",
    "{count} station(s) loaded.": "{count} station(s) chargée(s).",
    "Nothing imported": "Rien n'a été importé",
    "No rows were found.\n\nExpected columns: station, fov  "
    "(or any two columns: first = station, second = FOV).":
        "Aucune ligne trouvée.\n\nColonnes attendues : station, fov  "
        "(ou deux colonnes quelconques : la première = station, la seconde = champ de vision).",
    "Could not read CSV:\n{error}": "Impossible de lire le CSV :\n{error}",

    # Tags dialog
    "Tag Title": "Titre de l'étiquette",
    "Data Type": "Type de données",
    "Predefined Values (comma-separated)": "Valeurs prédéfinies (séparées par des virgules)",
    "Add Tag": "Ajouter une étiquette",
    "Remove Selected": "Supprimer la sélection",
    "Save": "Enregistrer",

    # Extensions dialog
    "Model Extensions": "Extensions de modèles",
    "Search models by name, region, author, task or species…":
        "Rechercher un modèle par nom, région, auteur, tâche ou espèce…",
    "Activity log …": "Journal d'activité …",
    "Fetching registry …": "Récupération du registre …",
    "Available": "Disponibles",
    "⚠  Could not reach registry: {error}": "⚠  Impossible d'accéder au registre : {error}",
    "No models found in registry.": "Aucun modèle trouvé dans le registre.",
    "{shown} of {total} model(s) match.": "{shown} modèle(s) sur {total} correspond(ent).",
    "{total} model(s) available.": "{total} modèle(s) disponible(s).",
    "{size} MB": "{size} Mo",
    "Author: {author}": "Auteur : {author}",
    "License: {license}": "Licence : {license}",
    "Citation": "Citation",
    "Info page": "Page d'information",
    "Classes: {classes}": "Classes : {classes}",
    "Installed ✓": "Installé ✓",
    "Download": "Télécharger",
    " Downloading …": " Téléchargement …",
    "{done:.1f} / {total:.1f} MB  (%p%)": "{done:.1f} / {total:.1f} Mo  (%p%)",
    "{done:.1f} MB …": "{done:.1f} Mo …",
    "Installed": "Installés",
    "Installed ({count})": "Installés ({count})",
    "Delete selected": "Supprimer la sélection",
    "  [bundled]": "  [intégrée]",
    "Bundled extension": "Extension intégrée",
    "This extension is bundled with the app and cannot be removed.":
        "Cette extension est intégrée à l'application et ne peut pas être supprimée.",
    "Delete extension": "Supprimer l'extension",
    "Remove '{name}' and all its files from disk?\nThis cannot be undone.":
        "Supprimer « {name} » et tous ses fichiers du disque ?\nCette action est irréversible.",
    "Removed {name}.": "{name} supprimé.",
    "Could not remove {name} — directory not found.": "Impossible de supprimer {name} : dossier introuvable.",
    "Publish a Model Extension": "Publier une extension de modèle",
    "Close": "Fermer",
}


def tr(text: str) -> str:
    if LANGUAGE == "fr":
        return FRENCH.get(text, text)
    return text


def translate_ui(root: QWidget) -> None:
    """Translate the static texts of *root* and every widget and action inside it.

    Only texts found in the dictionary change, so user data shown in labels is safe.
    Combo box items are left alone: several store their text as a saved value.
    """
    if LANGUAGE == "en":
        return
    for w in [root] + root.findChildren(QWidget):
        if w.isWindow() and w.windowTitle():
            w.setWindowTitle(tr(w.windowTitle()))
        if w.toolTip():
            w.setToolTip(tr(w.toolTip()))
        if isinstance(w, (QAbstractButton, QLabel)) and w.text():
            w.setText(tr(w.text()))
        if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit)) and w.placeholderText():
            w.setPlaceholderText(tr(w.placeholderText()))
        if isinstance(w, QGroupBox):
            w.setTitle(tr(w.title()))
        if isinstance(w, QMenu):
            w.setTitle(tr(w.title()))
        if isinstance(w, QTabWidget):
            for i in range(w.count()):
                w.setTabText(i, tr(w.tabText(i)))
        if isinstance(w, QTableWidget):
            for i in range(w.columnCount()):
                item = w.horizontalHeaderItem(i)
                if item:
                    item.setText(tr(item.text()))
    for action in root.findChildren(QAction):
        tip_is_text = action.toolTip() == action.text()
        action.setText(tr(action.text()))
        if not tip_is_text:
            action.setToolTip(tr(action.toolTip()))
