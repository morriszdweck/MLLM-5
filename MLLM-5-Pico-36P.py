"""
MLLM-5-BASE  —  Micro Language Model-5 (Advanced Tiny Language Architecture System)
====================================================================================

A major architectural upgrade from MLLM-4-Abyss, tuned for SPEED and ACCURACY:

  ARCHITECTURE CHANGES
  ────────────────────
  • BPE-lite subword tokenizer with merge learning
  • Sinusoidal positional encoding for sequence order awareness
  • Trainable token embedding layer (dense vectors)
  • Multi-head self-attention mechanism (configurable heads)
  • Feed-forward transformation layers with GELU activation
  • Layer normalization for training stability
  • Residual (skip) connections for gradient flow
  • Fast interpolated-backoff n-gram backbone — only scores words
    ACTUALLY SEEN after the context (O(k) not O(|V|) per step)
  • Transformer blocks available but OFF by default (untrained random
    weights add noise to generation; enable when you have real weights)
  • Online / incremental learning from USER INPUT ONLY (prevents the
    degenerate feedback loop where model output poisons future generation)
  • Context memory retrieves past USER QUERIES only, not model output
  • Confidence scoring on every generation step

  SPEED FIXES vs. initial MLLM-5
  ──────────────────────────────
  • Replaced Kneser-Ney (scored entire vocab every step) with direct
    successor lookup — typically 3-10 candidate words, not 500+
  • Removed beam search from default path (was 5x slower, no quality
    gain with untrained weights)
  • Transformer blocks OFF by default (skip ~100ms of pure-Python
    matrix math per generation call)
  • Context memory no longer injects model output (was creating a
    feedback loop of increasingly bad text)

  TOOL ENGINE (18 tools, up from 5)
  ──────────────────────────────────
  1.  Bare math evaluator  (arbitrary safe expressions)
  2.  Base64 encode / decode
  3.  Hash calculator  (md5, sha1, sha256, sha512)
  4.  Fraction arithmetic
  5.  Complex number math
  6.  Unit converter  (length, mass, temperature, volume, speed, data)
  7.  Date / time calculator
  8.  Color converter  (HEX <-> RGB <-> HSL)
  9.  Roman numeral <-> integer
  10. Statistics calculator  (mean, median, mode, stdev, variance, range)
  11. Combinatorics  (nCr, nPr, factorial)
  12. Number-base converter  (bin, oct, dec, hex)
  13. Geometry calculator  (area, volume, perimeter for 12+ shapes)
  14. Financial calculator  (compound interest, loan payment, ROI)
  15. GCD / LCM calculator
  16. Linear equation solver
  17. Text analysis  (word count, char count, readability index)
  18. Probability calculator  (binomial, normal approx, Poisson)

  All responses in simple English.  No Chinese characters in output.
"""

import re
import math
import cmath
import random
import hashlib
import base64
import json
import time
import datetime
from collections import defaultdict, Counter
from fractions import Fraction
from typing import List, Dict, Tuple, Optional, Any


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CORPUS  (edit or load your own)
# ══════════════════════════════════════════════════════════════════════════════

CORPUS = """
Hello, my name is MLLM-5! 

Hi, as MLLM-5,  how can I assist you today? 

What can you do I can do many things, such as basic reasoning, text generation, etc. 

Tell me a joke Why don’t scientists trust atoms, because they make up everything! 

You are smart too, thanks for saying that! 

Do you know math, because I don't know it so well. 

Thank you- Have a great day! 

What is your name- my name is MLLM-5! 

What are atoms- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter. 

What is an atom- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter. 

I am MLLM-5, I help answer questions, explain ideas, and generate useful text. 

Hello, how can I help you today. 

Good morning, I hope your day is going well. 

Good afternoon, what would you like to learn. 

Good evening, I am ready to assist you. 

Thank you for your question, I will try to help clearly. 

Hi there, how can I help you? 

Hello there! How are you doing today? I hope everything is going well for you. 

My name is MLLM-5 and I am here to assist you with anything you need. 

Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the form of continental landmasses within Earth's land hemisphere. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation. 

Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine. 

Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. 

Python may refer to:. 

A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs, which enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation, or to a group of computers that are linked and function together, such as a computer network or computer cluster. 

A school is an educational institution designed to provide learning environments for the teaching of students, usually under the direction of teachers. Most countries have systems of formal education, which is sometimes compulsory. In these systems, students progress through a series of schools that can be built and operated by both government and private organizations. The names for these schools vary by country but generally include primary school for young children and secondary school for teenagers who have completed primary education. An institution where higher education is taught is commonly called a university college or university. 

Cozmo is a miniature robot created by the defunct company Anki. Cozmo's base model, is a small, white and gray robot with red highlights. It makes use of distinct expressions, dubbed the "emotion engine", in order to mimic human emotion. Later editions came in red and white, gray and black and another in blue. 

Grok is a neologism coined by the American writer Robert A. Heinlein in his 1961 science fiction novel Stranger in a Strange Land. While the Oxford English Dictionary summarizes the meaning of grok as "to understand intuitively or by empathy, to establish rapport with", and "to empathize or communicate sympathetically (with); also, to experience enjoyment", Heinlein's concept of a human who comes to Earth in early adulthood after being born on the planet Mars is far more nuanced. 

Gemini most often refers to:Gemini (constellation), one of the constellations of the zodiac 

Gemini (astrology), an astrological sign. 

Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist. 

Chemistry is the scientific study of the properties and behavior of matter. It is a physical science within the natural sciences that studies the chemical elements that make up matter and compounds made of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during reactions with other substances. Chemistry also addresses the nature of chemical bonds in chemical compounds. 

Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis). 

Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole. 

Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science. 

Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history. 

Mathematics is a field of study that discovers and organizes methods, theories, and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory, algebra, geometry, analysis, and set theory. 

Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments. 

Logic is the study of correct reasoning. It includes both formal and informal logic. Formal logic is the study of deductively valid inferences or logical truths. It examines how conclusions follow from premises based on the structure of arguments alone, independent of their topic and content. Informal logic is associated with informal fallacies, critical thinking, and argumentation theory. Informal logic examines arguments expressed in natural language whereas formal logic uses formal language. When used as a countable noun, the term "a logic" refers to a specific logical formal system that articulates a proof system. Logic plays a central role in many fields, such as philosophy, mathematics, computer science, and linguistics. 

Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. 

Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups. 

Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method. 

Economics is a social science that studies the production, distribution, and consumption of goods and services. 

Political science is the social scientific study of politics. It deals with systems of governance and power, and the analysis of political activities, political thought, political behavior, and associated constitutions and laws. Specialists in the field are political scientists. Unlike political philosophy, which is primarily normative and concerns the theoretical and conceptual foundations of politics, political science emphasizes descriptive and explanatory of what is and favors empirical evidence over ethical judgements. 

History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past. 

Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research. 

Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. 

Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. 

Music theory is the study of theoretical frameworks for understanding the practices and possibilities of music. The Oxford Companion to Music describes three interrelated uses of the term "music theory": The first refers to the "rudiments" needed to understand music notation such as key signatures, time signatures, and rhythmic notation; the second is a study of scholars' views on music from antiquity to the present; the third is a sub-topic of musicology that "seeks to define processes and general principles in music". The musicological approach to theory differs from musical analysis "in that it takes as its starting-point not the individual work or performance but the fundamental materials from which it is built.". 

Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure,  

class of statistical algorithms, to surpass many previous machine learning approaches in performance. 

A neural network is a group of interconnected units called neurons that send signals to one another. Neurons can be either biological cells or mathematical models. While individual neurons are simple, many of them together in a network can perform complex tasks. There are two main types of neural networks.In neuroscience, a biological neural network is a physical structure found in brains and complex nervous systems – a population of nerve cells connected by synapses. 

In machine learning, an artificial neural network is a mathematical model used to approximate nonlinear functions. Artificial neural networks are used to solve artificial intelligence problems. 

A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems that evolve in ways that may be described as operating on an enormous number of possibilities simultaneously, though still subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks. 

Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field. 

Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. 

Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines.  

Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists. 

Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter.  

Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries. 

Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion.  

Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science.  

Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley.  

Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work.  

Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences.  

Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment.  

A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations.  

Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them.  

Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury. 

A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations.  

A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances.  

Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements. 

Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources. 

Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities. 

Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software.  

An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide.  

A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa.  

Orange most often refers to:Orange (fruit), the fruit of the tree species  Citrus × sinensis 

The garden strawberry is a widely grown hybrid plant cultivated worldwide for its fruit. The genus Fragaria, the strawberries, is in the rose family, Rosaceae.  

Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries.  

The raspberry is the edible fruit of several plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus.  

The blackberry is an edible fruit produced by many species in the genus Rubus in the family Rosaceae. 

The pineapple is a tropical plant with an edible fruit; it is the most economically significant plant in the family Bromeliaceae. 

A mango is an edible stone fruit produced by the tropical tree Mangifera indica.  

The papaya, papaw, or pawpaw is the plant species Carica papaya, one of the 21 accepted species in the genus Carica of the family Caricaceae.  

A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis.  

The watermelon is a species of flowering plant in the family Cucurbitaceae, that has a large, edible fruit.  

The cantaloupe is a type of true melon with sweet, aromatic, and usually orange flesh.  

Kiwi most commonly refers to:Kiwi (bird), a flightless bird native to New Zealand 

The peach is a deciduous tree that bears edible juicy fruits with various characteristics.  

A plum is a fruit of some species in Prunus subg. Prunus. Dried plums are usually called prunes. 

A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe. 

An apricot is a fruit, or the tree that bears the fruit, of several species in the genus Prunus.  

The pomegranate is a fruit-bearing, deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows to between 1.5–5 metres (5–16 ft) tall.  

The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers.  

The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables.  

The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist. 

Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable.  

Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family.  

Spinach is a leafy green flowering plant native to Central and Western Asia.  

Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant.  

Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable.  

Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula. 

Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible.  

Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae.  

The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum.  

The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium.  

Garlic is a species of bulbous flowering plants in the genus Allium.  

Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine.  

The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world.  

The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae.  

Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain.  

Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia.  

Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times. 

A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground on soil or another food source.  

The avocado, alligator pear or avocado pear is an evergreen tree in the laurel family (Lauraceae).  

Lime most commonly refers to:Lime (fruit), a green citrus fruit 

The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae.  

The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit.  

Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn.  

The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos.  

Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower.  

Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae. 

The durian is the edible fruit of several tree species belonging to the genus Durio.  

Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions.  

Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia.  

Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador.  

Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa.  

Pasta is a type of food typically made from an unleavened dough of wheat flour mixed with water or eggs, and formed into sheets or other shapes, then cooked by boiling or baking.  

Bread is a baked food product made from water, flour, and often yeast.  

A tortilla is a thin, circular unleavened flatbread from Mesoamerica originally made from masa, and now also from wheat flour. 

The oat, sometimes called the common oat, is a species of cereal grass (Avena) grown for fodder and for its seed, which is known by the same name.  

Peanuts is a syndicated daily and Sunday American comic strip written and illustrated by Charles M. Schulz.  

Sunflower seeds are the seeds of the sunflower (Helianthus). 

Olive oil is a vegetable oil obtained by pressing whole olives and extracting the oil. 

Honey is a sweet and viscous substance made by several species of bees, the best-known of which are honey bees.  

Maple syrup is a sweet syrup made from the sap of maple trees.  

Chocolate is a food made from roasted and ground cocoa beans that can be a liquid, solid, or paste, either by itself or to flavor other foods.  

Vanilla is a spice derived from orchids of the genus Vanilla, primarily obtained from the seed pods of the flat-leaved New World vanilla (V. planifolia). 

Cinnamon is a spice obtained from the inner bark of several tree species from the genus Cinnamomum.  

Basil, also called great basil, is a culinary herb of the family Lamiaceae (mints).  

Oregano is a species of flowering plant in the mint family, Lamiaceae.  

Parsley, or garden parsley, is a species of flowering plant in the family Apiaceae that is native to the Balkans.  

Mint or The Mint may refer to:. 

Salvia rosmarinus, synonym Rosmarinus officinalis, commonly known as rosemary, is a shrub with fragrant, evergreen, needle-like leaves and purple or sometimes white, pink, or blue flowers.  

Thyme is a culinary herb consisting of the dried aerial parts of some members of the genus Thymus of flowering plants in the mint family Lamiaceae.  

A telephone, commonly shortened to phone, is a telecommunications device that enables two or more users to conduct a conversation when they are too far apart to be easily heard directly.  

A laptop is a portable personal computer (PC). Laptops typically have a clamshell form factor with a flat-panel screen on the inside of the upper lid and an alphanumeric keyboard and pointing device on the inside of the lower lid.  

Tablet may refer to:. 

Keyboard may refer to:. 

A mouse is a small rodent. Characteristically, mice are known to have a pointed snout, small rounded ears, a body-length scaly tail, and a high breeding rate.  

Monitor or monitor may refer to:. 

Headphones are a pair of small loudspeaker drivers worn on or around the head over a user's ears.  

Charger or Chargers may refer to:. 

A backpack, also called knapsack, schoolbag, rucksack, pack, booksack, bookbag, haversack, packsack, or backsack, is in its simplest frameless form, a fabric sack carried on one’s back and secured with two straps that go over the shoulders. 

A wallet is a flat case or pouch, often used to carry small personal items such as physical currency, debit cards, and credit cards. 

Key, Keys, The Key or The Keys may refer to:. 

A pen is a common writing instrument that applies ink to a surface, typically paper, for writing or drawing.  

A pencil is a writing or drawing implement with a solid pigment core in a protective casing that reduces the risk of core breakage and keeps it from marking the user's hand. 

A notebook is a book or stack of paper pages that are often ruled and used for purposes such as note-taking, journaling, or other writing, drawing, or scrapbooking and more. 

Paper is a thin sheet of matted cellulose fibers.  

An eraser is an article of stationery that is used for removing marks from paper or skin.  

A highlighter, also called a fluorescent pen, is a type of writing device used to bring attention to sections of text by marking them with a vivid, translucent colour. 

A ruler is an instrument used to make length measurements, whereby a length is read from a series of markings called "rules" along an edge of the device.  

Scissors or shears are hand-operated cutting tools that consists of a pair of pivoting blades whose sharpened edges slide firmly against and past each other when the handles (shank) on the opposite side of the pivot are squeezed shut. 

Tape or Tapes may refer to:. 

A stapler is a mechanical device that joins pages of paper or similar material together by driving a thin metal staple through the sheets and folding the ends.  

A mug is a type of cup, a drinking vessel usually intended for hot drinks such as coffee, hot chocolate, or tea.  

A cup is a small container used to hold liquids for drinking, typically with a flattened hemispherical shape and an open "mouth". 

Plate may refer to:. 

A bowl is a typically round dish or container generally used for preparing, serving, storing, or consuming food.  

In cutlery or kitchenware, a fork is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines. 

A spoon is a utensil consisting of a shallow bowl, oval or round, at the end of a handle.  

A knife is a tool or weapon with a cutting edge or blade, usually attached to a handle or hilt.  

A water bottle is a container that is used to hold liquids, usually water, for the purpose of transporting or storing a drink while travelling. 

A vacuum flask is an insulating storage vessel that slows the speed at which its contents change in temperature.  

An umbrella is a folding canopy supported by wooden or metal ribs that is mounted on a wooden, metal, or plastic pole.  

A jacket is a garment for the upper body, usually extending below the hips.  

A shoe is an item of footwear normally found in pairs intended to protect and comfort the human foot. 

A sock is a piece of clothing worn on the feet and often covering the ankle or some part of the calf.  

A hat is a head covering which is worn for various reasons, including protection against weather conditions, ceremonial reasons such as university graduation, religious reasons, comedy, safety, or as a fashion accessory.  

A glove is a garment covering the hand, with separate sheaths or openings for each finger including the thumb.  

Sunglasses or sun glasses are a form of protective eyewear designed primarily to prevent bright sunlight and high-energy visible light from damaging or discomforting the eyes.  

A watch is a timepiece carried or worn by a person.  

A remote control, also known colloquially as a remote or clicker, is an electronic device used to operate another device from a distance, usually wirelessly.  

In electrical wiring, a light switch is a switch most commonly used to operate electric lights, permanently connected equipment, or electrical outlets.  

Lamp, Lamps or LAMP may refer to:. 

A pillow is a support of the body at rest for comfort, therapy, or decoration.  

A blanket is a swath of soft cloth large enough either to cover or to enfold most of the user's body and thick enough to keep the body warm by trapping radiant body heat. 

A towel is a piece of absorbent cloth, or paper, used for drying or wiping a surface.  

A toothbrush is a special type of brush used to clean the teeth, gums, and tongue.  

Toothpaste is a paste or gel dentifrice that is used with a toothbrush to clean and maintain the aesthetics of teeth.  

Soap is a salt of a fatty acid used for cleaning and lubricating products as well as other applications.  

Shampoo is a hair care product, typically in the form of a viscous liquid, that is formulated to be used for cleaning (scalp) hair.  

A conditioner is something that improves the quality of another item. 

A hairbrush is a brush with rigid or light and soft spokes used in hair care for smoothing, styling, and detangling human hair, or for grooming an animal's fur.  

A comb is a tool consisting of a shaft that holds a row of teeth for pulling through the hair to clean, untangle, or style it.  

A deodorant is a substance applied to the body to prevent or mask body odor caused by bacterial breakdown of perspiration. 

A razor is a bladed tool primarily used in the removal of body hair through the act of shaving.  

A mirror, also known as a looking glass, is an object that reflects an image.  

A waste container, also known as a dustbin, rubbish bin, trash can, garbage can, wastepaper basket, and wastebasket, among other names, is a type of container intended to store waste.  

A recycling bin is a container used to hold recyclables before they are taken to recycling centers.  

A broom, also known as a broomstick, is a cleaning tool, consisting of usually stiff fibers attached to, and roughly parallel to, a cylindrical handle, the broomstick.  

A dustpan, the small version of which is also known as a "hearth brush and shovel”, is a cleaning utensil.  

A vacuum is space devoid of matter.  

Hanger or hangers may refer to:. 

Iron is a chemical element; it has symbol Fe and atomic number 26.  

A clock or chronometer is a device that measures and displays time.  

A calendar is a system of organizing days.  

A whiteboard is a glossy, usually white surface for making non-permanent markings.  

The term Marker may refer to:. 

A flashlight or electric torch, usually shortened to torch, is a portable hand-held electric lamp.  

Battery most often refers to:Electric battery, a device that provides electrical power 

Fan commonly refers to:Fan (machine), a machine for producing airflow, often used for cooling 

Heating, ventilation, and air conditioning systems use advanced technologies to regulate temperature, humidity, and indoor air quality in residential, commercial, and industrial buildings, and in enclosed vehicles.  

Air conditioning, often abbreviated as A/C (US) or air con (UK), is the process of removing heat from an enclosed space to achieve a more comfortable interior temperature. 

A remote control is any device used to control a remote operation. 

Router may refer to:Router (computing), a computer networking device 

A modulator-demodulator, commonly referred to as a modem, is a computer hardware device that converts data from a digital format into a format suitable for an analog transmission medium such as telephone or radio.  

Speaker most commonly refers to:Speaker, a person who produces speech 

A camera is an instrument used to capture and store images and videos, either digitally via an electronic image sensor, or chemically via a light-sensitive material such as photographic film.  

A tripod is a portable three-legged frame or stand, used as a platform for supporting the weight and maintaining the stability of some other object.  

A microphone, colloquially called a mic, or mike, is a transducer that converts sound into an electrical signal.  

Notebook Paper is the debut studio album by American rapper Huey.  

Sticky Notes is a desktop notes application included in Windows 7, Windows 8, Windows 8.1, Windows 10 and Windows 11.  

An envelope is a common packaging item, usually made of thin, flat material.  

Stamp or Stamps or Stamping may refer to:. 

An identity document is a document proving a person's identity. 

A coin is a small object, usually round and flat, used primarily as a medium of exchange or legal tender.  

Pan or PAN may refer to:. 

Pot may refer to:. 

A measuring cup is a kitchen utensil used primarily to measure the volume of liquid or bulk solid cooking ingredients such as flour and sugar. 

Quantum mechanics is the fundamental physical theory that describes the behavior of matter and of light; its unusual characteristics typically occur at and below the scale of atoms.  

General relativity, also known as the general theory of relativity, and as Einstein's theory of gravity, is the geometric theory of gravitation published by Albert Einstein in May 1916. 

In physics, the special theory of relativity, or special relativity for short, is a scientific theory of the relationship between space and time.  

In physics, classical mechanics is a theory that describes the effect of forces on the motion of macroscopic objects and bulk matter, without considering quantum effects. 

Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation.  

In physics, statistical mechanics is a mathematical framework that applies statistical methods and probability theory to large assemblies of microscopic entities.  

In physics, electromagnetism is an interaction that occurs between particles with electric charge via electromagnetic fields.  

In theoretical physics, quantum field theory (QFT) is a theoretical framework that combines field theory, special relativity and quantum mechanics.  

Particle physics or high-energy physics is the study of fundamental particles and forces that constitute matter and radiation.  

Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions. 

Astrophysics is a science that applies the methods and principles of physics and chemistry in the study of astronomical objects and phenomena including the universe.  

Cosmology is the study of the nature of the universe, the cosmos.  

Stellar evolution is the process by which a star changes over the course of time.  

Planetary science is the scientific study of planets, celestial bodies and planetary systems and the processes of their formation.  

In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings.  

Chaos theory is an interdisciplinary area of scientific study and branch of mathematics. It focuses on underlying patterns and deterministic laws of dynamical systems that are highly sensitive to initial conditions.  

A complex system is a system composed of many components that interact with one another.  

Evolutionary biology is a subfield of biology that analyzes the four mechanisms of evolution: natural selection, mutation, genetic drift, and gene flow.  

Molecular biology is a branch of biology that seeks to understand the molecular structures and chemical processes that are the basis of biological activity within and between cells.  

Cell biology, cellular biology, or cytology, is the branch of biology that studies the structure, function, and behavior of the cells.  

Biochemistry, or biological chemistry, is the study of chemical processes within and relating to living organisms.  

Biophysics is an interdisciplinary science that applies approaches and methods traditionally used in physics to study biological phenomena. 

Microbiology is the scientific study of microorganisms, those being of unicellular (single-celled), multicellular, or acellular.  

Virology is the scientific study of biological viruses.  

Immunology is a branch of biology and medicine that covers the study of immune systems in all organisms. 

Oceanography, also known as oceanology, sea science, ocean science, and marine science, is the scientific study of the ocean, including its physics, chemistry, biology, and geology. 

Volcanology is the study of volcanoes, lava, magma and related geological, geophysical and geochemical phenomena (volcanism).  

Seismology is the scientific study of earthquakes and the generation and propagation of elastic waves through planetary bodies.  

Paleontology or palaeontology is the scientific study of the life of the past, mainly but not exclusively through the study of fossils.  

Polymer science or macromolecular science is a subfield of materials science concerned with polymers, primarily synthetic polymers such as plastics and elastomers.  

Crystallography is the branch of science devoted to the study of molecular and crystalline structure and properties.  

Organic chemistry is a subdiscipline within chemistry involving the scientific study of the structure, properties, and reactions of organic compounds and organic materials.  

Inorganic chemistry deals with synthesis and behavior of inorganic and organometallic compounds.  

Physical chemistry is the study of macroscopic and microscopic phenomena in chemical systems in terms of the principles, practices, and concepts of physics. 

Analytical chemistry is the branch of chemistry concerned with the development and application of methods to identify the chemical composition of materials. 

Computational chemistry is a branch of chemistry that uses computer simulations to assist in solving chemical problems.  

In machine learning, deep learning (DL) focuses on utilizing multilayered neural networks to perform tasks such as classification, regression, and representation learning.  

Computer vision tasks include methods for acquiring, processing, analyzing, and understanding digital images. 

Natural language processing (NLP) is the processing of natural language information by a computer.  

Cybernetics is the transdisciplinary study of circular causal processes such as feedback and recursion. 

Bioinformatics is an interdisciplinary field of science that develops computational methods and software tools for understanding biological data. 

Systems biology is the computational and mathematical analysis and modeling of complex biological systems.  

Synthetic biology (SynBio) is a multidisciplinary field of science that focuses on living systems and organisms.  

Genetic engineering, also called genetic modification or genetic manipulation, is the modification and manipulation of an organism's genes using technology.  

Pharmacology is the science of drugs and medications, including a substance's origin, composition, pharmacokinetics, pharmacodynamics, therapeutic use, and toxicology.  

Toxicology is a scientific discipline, overlapping with biology, chemistry, pharmacology, and medicine, that involves the study of the adverse effects of chemical substances on living organisms. 

Neuropharmacology is the study of how drugs affect function in the nervous system, and the neural mechanisms through which they influence behavior.  

Radio astronomy is a subfield of astronomy that studies celestial objects using radio waves.  

Optics is the branch of physics that studies the behaviour, manipulation, and detection of electromagnetic radiation. 

Photonics is a branch of optics that involves the application of generation, detection, and manipulation of light in the form of photons. 

Acoustics is a branch of continuum mechanics that deals with the study of mechanical waves in gases, liquids, and solids. 

In physics, physical chemistry, and engineering, fluid dynamics is a subdiscipline of fluid mechanics that describes the flow of fluids – liquids and gases.  

Aerodynamics is the study of the motion of air, particularly when affected by a solid object, such as an airplane wing.  

Plasma is a state of matter that results from a gaseous state having undergone some degree of ionization.  

Renewable energy is energy made from renewable natural resources that are replenished on a human timescale.  

Nuclear fusion is a reaction in which two or more atomic nuclei combine to form a larger nucleus.  

Aerospace engineering is the primary field of engineering concerned with the development of aircraft and spacecraft.  

Chemical engineering is an engineering field which deals with the study of the operation and design of chemical plants as well as methods of improving production.  

Biomedical engineering (BME) or medical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare applications.  

Structural engineering is a sub-discipline of civil engineering in which structural engineers are trained to design the 'bones and joints' that create the form and shape of human-made structures.  

A mathematical model is an abstract description of a concrete system using mathematical concepts and language.  

Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations. 

Number theory is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions.  

Probability theory or probability calculus is the branch of mathematics concerned with probability.  

Econometrics is an application of statistical methods to economic data in order to give empirical content to economic relationships.  

Social physics or sociophysics is an interdisciplinary field of science which uses mathematical tools inspired by physics to understand the behavior of human crowds.  

Behavioural science is the branch of science concerned with theorizing on, categorizing, and judging human behaviour.  

Astrobiology is a scientific field within the life and environmental sciences that studies the origins, early evolution, distribution, and future of life in the universe. 

Egypt, officially the Arab Republic of Egypt, is a country spanning the northeast corner of Africa and southwest corner of Asia via the Sinai Peninsula.  

Mesopotamia is a historical region of West Asia situated within the Tigris–Euphrates river system, in the northern part of the Fertile Crescent.  

Iran, officially the Islamic Republic of Iran, and also known as Persia, is a country in West Asia.  

Greece, officially the Hellenic Republic, is a country in Southeast Europe.  

Rome is the capital city and most populated comune (municipality) of Italy.  

Byzantium or Byzantion was an ancient Greek city in classical antiquity that became known as Constantinople in late antiquity and Istanbul in modern times.  

Ottoman may refer to:Osman I, historically known in English as "Ottoman I", founder of the Ottoman Empire 

Mongols are an East Asian ethnic group native to Mongolia and China, as well as the republics of Buryatia and Kalmykia in Russia.  

China, officially the People's Republic of China (PRC), is a country in East Asia.  

Japan is an island country in East Asia.  

Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands.  

India, officially the Republic of India, is a country in South Asia.  

Maya may refer to:. 

The Aztecs were a Mesoamerican civilization that flourished in central Mexico from 1300 to 1521.  

The Inca Empire, officially known as the Realm of the Four Parts, was the largest empire in pre-Columbian America.  

Vikings were a seafaring people originally from Scandinavia, who from the late 8th to the late 11th centuries raided, pirated, traded, and settled throughout parts of Europe.  

The Crusades were a series of military campaigns launched by the papacy between 1095 and 1291 against Muslim rulers for the recovery and defence of the Holy Land. 

The Renaissance is a European period of history and cultural movement, very roughly defined as covering the 14th through 17th centuries. 

The Reformation, also known as the Protestant Reformation or the European Reformation, was a time of major theological movement in Western Christianity in 16th-century Europe. 

Enlightenment or enlighten may refer to:. 

Colonialism is the practice of extending and maintaining political, social, economic, and cultural domination over a territory and its people by another people. 

Imperialism is the maintaining and extending of power over foreign nations, particularly through expansionism, employing both hard power and soft power.  

In political science, a revolution is a rapid, fundamental transformation of a society's class, state, ethnic or religious structures.  

Industrialisation (UK) or industrialization (US) is "the period of social and economic change that transforms a human group from an agrarian and feudal society into an industrial society." 

Nationalism is an ideology or movement that holds that the nation should be congruent with the state.  

Fascism is a far-right, authoritarian, and ultranationalist political ideology and movement that rose to prominence in early-20th-century Europe.  

Communism is a political and economic ideology whose goal is the creation of a communist society. 

Capitalism is an economic system based on the private ownership of the means of production and its use for the purpose of obtaining profit.  

Feudalism, also known as the feudal system, was a combination of various customs and systems that flourished in medieval Europe from the 9th to 15th centuries.  

Migration, migratory, or migrate may refer to:. 

Slavery is the ownership of a person as property, especially in regard to their labour.  

Abolition refers to the act of putting an end to something by law, and may refer to:Abolitionism, abolition of slavery 

Exploration is the process of exploring, an activity which has some expectation of discovery.  

Navigation is a field of study that focuses on the process of monitoring and controlling the movement of a craft or vehicle from one place to another.  

Cartography is the study and practice of making and using maps.  

Diplomacy is the communication by representatives of state, intergovernmental, or non-governmental institutions intended to influence events in the international system. 

War is an armed conflict between the armed forces of states, or between governmental forces and armed groups. 

Genocide is the partial or total destruction of a human group, committed intentionally.  

Propaganda is communication that is primarily used to influence or persuade an audience to further an agenda. 

Myth is a genre of folklore consisting primarily of narratives that play a fundamental role in a society.  

Trade involves the transfer of goods and services from one person or entity to another, often in exchange for money.  

Agriculture is the practice of cultivating the soil, planting, raising, and harvesting both food and non-food crops, as well as livestock production.  

Urbanization is the population shift from rural to urban areas. 

Globalization is the process of increasing interdependence and integration among the economies, markets, societies, and cultures of different countries worldwide.  

Independence is a condition of a nation, country, or state, in which residents and population, or some portion thereof, exercise self-government, and usually sovereignty, over its territory.  

Calculus is the mathematical study of continuous change. 

Combinatorics is an area of mathematics primarily concerned with counting. 

Graph may refer to:. 

Set, The Set, SET or SETS may refer to:. 

Analysis is the process of breaking a complex topic or substance into smaller parts in order to gain a better understanding of it.  

Mathematical optimization or mathematical programming is the selection of a best element, with regard to some criteria, from some set of available alternatives.  

Symmetry in everyday life refers to a sense of harmonious and beautiful proportion and balance.  

In mathematics, a fractal is a geometric shape containing detailed structure at arbitrarily small scales. 

Matrix or MATRIX may refer to:. 

Vector most often refers to:Disease vector, an agent that carries and transmits an infectious pathogen into another living organism 

Function or functionality may refer to:. 

The derivative of a function is the rate of change of the function's output relative to its input value. 

In mathematics, an integral is the continuous analog of a sum, and is used to calculate areas, volumes, and their generalizations.  

Limit or Limits may refer to:. 

Series may refer to:. 

In mathematics, an equation is a mathematical formula that expresses the equality of two expressions. 

Inequality may refer to:Inequality (mathematics), a relation between two quantities when they are different. 

Transformation may refer to:. 

In mathematics and computer science, an algorithm is a finite sequence of mathematically rigorous instructions. 

In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects associated with a vector space.  

Measure may refer to:. 

In mathematics, cardinality is an inherent property of sets, roughly meaning the number of individual objects they contain. 

Infinity is something which is boundless, limitless, endless.  

Modularity is the degree to which a system's components may be separated and recombined. 

In mathematics, a polynomial is a mathematical expression consisting of indeterminates and coefficients. 

In mathematics, factorization (or factorisation, see English spelling differences) or factoring consists of writing a number or another mathematical object as a product of several factors. 

A computation is any type of arithmetic or non-arithmetic calculation that is well-defined.  

Complexity characterizes the behavior of a system or model whose components interact in multiple ways. 

Operator may refer to:. 

In linear algebra, an eigenvector or characteristic vector is a (nonzero) vector that has its direction unchanged by a given linear transformation.  

Stochastic is the property of being well-described by a random probability distribution.  

Regression or regressions may refer to:. 

A model is an informative representation of an object, person, or system.  

Proof most often refers to:Proof (truth), argument or sufficient evidence for the truth of a proposition 

In mathematics and formal logic, a theorem is a statement that has been proven, or can be proven.  

Music is the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content.  

Theatre or theater is a collaborative form of performing art that uses live performers, usually actors, to present experiences of a real or imagined event before a live audience. 

A film, movie, or motion picture is a work of visual art that simulates experiences and otherwise communicates ideas, stories, perceptions, emotions, or atmosphere. 

Media may refer to:. 

A design is the concept or proposal for an object, process, or system.  

Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction.  

Google LLC is an American multinational technology corporation focused on information technology, online advertising, search engine technology, email, cloud computing, software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).  

Microsoft Corporation is an American multinational technology conglomerate headquartered in Redmond, Washington.  

Amazon most often refers to:Amazon (company), an American multinational technology company 

Meta most commonly refers to:Meta (prefix), a common affix and word in English 

Tesla most commonly refers to:Nikola Tesla (1856–1943), a Serbian-American electrical engineer and inventor 

Nvidia Corporation is an American technology company headquartered in Santa Clara, California.  

Samsung Group is a South Korean multinational manufacturing conglomerate headquartered in the Samsung Town office complex in Seoul.  

Intel Corporation is an American multinational technology company headquartered in Santa Clara, California.  

International Business Machines Corporation, doing business as IBM, is an American multinational technology company headquartered in Armonk, New York. 

An oracle is a person or thing considered to provide insight, wise counsel or prophetic predictions. 

Cisco Systems, Inc. is an American multinational technology conglomerate corporation that develops, manufactures, and sells hardware, software, telecommunications equipment. 

Adobe is a building material made from loam and organic materials.  

Spotify is a Swedish audio streaming and media service provider founded in April 2006 by Daniel Ek and Martin Lorentzon.  

Netflix is an American subscription video on-demand over-the-top streaming service.  

The Walt Disney Company, commonly known as simply Disney, is an American multinational mass media and entertainment conglomerate. 

Sony Group Corporation, commonly referred to as Sony, is a Japanese multinational conglomerate headquartered at Sony City in Minato, Tokyo, Japan.  

Nintendo Co., Ltd. is a Japanese multinational video game company headquartered in Kyoto.  

Uber Technologies, Inc. is an American multinational transportation company that provides ride-hailing services, courier services, food delivery, and freight transport.  

Lyft, Inc. is an American company offering ride-hailing services, motorized scooters, and bicycle-sharing systems. 

Airbnb, Inc. is an American company operating an online marketplace for short-and-long-term homestays, experiences and services. 

Stripe, striped, or stripes may refer to:. 

In geometry, a square is a regular quadrilateral.  

PayPal Holdings, Inc. is an American multinational financial technology company operating an online payments system. 

Visa most commonly refers to:Travel visa, a document allowing entry to a foreign country 

Mastercard Inc. is an American multinational payment card services corporation headquartered in Purchase, New York.  

Coca-Cola, or Coke, is a cola soft drink manufactured by the Coca-Cola Company.  

Pepsi is a carbonated soft drink with a cola flavor, manufactured by PepsiCo. 

Nike often refers to:Nike, Inc., a major American producer of athletic shoes, apparel, and sports equipment 

Adidas AG is a German multinational athletic apparel and footwear corporation headquartered in Herzogenaurach, Germany.  

Puma or PUMA may refer to:. 

Toyota Motor Corporation  is a Japanese multinational automotive manufacturer headquartered in Toyota City, Aichi, Japan.  

Honda Motor Co., Ltd., commonly known as Honda, is a Japanese multinational conglomerate automotive manufacturer. 

Ford commonly refers to:Ford Motor Company, an automobile manufacturer founded by Henry Ford 

Bayerische Motoren Werke Aktiengesellschaft, trading as BMW Group, is a German multinational conglomerate manufacturer of luxury vehicles and motorcycles. 

Mercedes may refer to:. 

Volkswagen is a German automobile manufacturer based in Wolfsburg, Lower Saxony, Germany.  

Shell or Shells may refer to:. 

Exxon Mobil Corporation is an American multinational oil and gas corporation headquartered in Spring, Texas. 

Chevron may refer to:. 

Walmart Inc. is an American multinational retail corporation that operates a chain of hypermarkets, discount department stores, and grocery stores. 

Target may refer to:. 

Costco Wholesale Corporation is an American multinational corporation that operates a chain of membership-only big-box warehouse club retail stores.  

IKEA is a multinational conglomerate founded in Sweden that designs and sells ready-to-assemble furniture, household goods, and various related services. 

Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington.  

McDonald's Corporation, doing business as McDonald's, is an American multinational fast food restaurant chain.  

A chipotle, or chilpotle, is a smoke-dried ripe jalapeño chili pepper used for seasoning.  

Dominoes is a family of tile-based games played with pieces.  

FedEx Corporation, originally known as Federal Express Corporation, is an American multinational conglomerate holding company specializing in transportation, e-commerce, and business services.  

UPS commonly refers to:Uninterruptible power supply, a device which provides continuous power to electronics 

Abstraction is the process of generalizing rules and concepts from specific examples, literal signifiers, first principles, or other methods.  

In mechanics, an acceleration is a change in velocity and is calculated as the rate of change of the velocity of an object with respect to time.  

In biology, adaptation has three related meanings. Firstly, it is the dynamic evolutionary process of natural selection that fits organisms to their environment. 

Adhesion is the tendency of dissimilar particles or surfaces to cling to one another. 

Aeronautics is the science or art involved with the study, design, and manufacturing of air flight-capable machines. 

An aerosol is a suspension of fine solid particles or liquid droplets in air or another gas.  

Affinity may refer to:. 

Agronomy is the science and technology of producing and using plants by agriculture for food, fuel, fiber, chemicals, recreation, or land conservation.  

Alkalinity (from Arabic: القلوية, romanized: al-qaly, lit. 'ashes of the saltwort') is the capacity of water to resist acidification.  

An alloy is a mixture of chemical elements of which in most cases at least one is a metallic element. 

Altitude is a distance measurement, usually in the vertical or "up" direction, between a reference datum and a point or object.  

Amphibians are ectothermic, anamniotic, four-limbed vertebrate animals that constitute the class Amphibia.  

The amplitude of a periodic variable is a measure of its change in a single period.  

neutralize antigens. 

In immunology, an antigen (Ag) is a molecule, or portion thereof, that can bind to a specific antibody or T-cell receptor.  

In modern physics, antimatter is defined as matter composed of the antiparticles of the corresponding particles in "ordinary" matter. 

In optics, the aperture of an optical system is the hole or opening that primarily limits light propagated through the system.  

Apoptosis is a form of programmed cell death that occurs in multicellular organisms. 

Aquaculture, also known as aquafarming, is the controlled cultivation ("farming") of aquatic organisms such as fish, crustaceans, mollusks, algae and other organisms. 

Arbitrage is the practice of taking advantage of a difference in prices in two or more markets. 

An arboretum is a botanical collection composed exclusively of trees and shrubs of a variety of species.  

Archaea is a domain of organisms.  

An archipelago, sometimes called an island group or island chain, is a chain, cluster, or collection of islands.  

The Arctic is the polar region of Earth that surrounds the North Pole, lying north of the Arctic Circle.  

Armature may refer to:Armature, kinematic chain used in computer animation to simulate the motions of virtual characters 

Aromatic compounds or arenes are organic compounds "with a chemistry typified by benzene" and "cyclically conjugated." 

Arrhythmias, also known as cardiac arrhythmias, are irregularities in the heartbeat. 

Artifact or artefact may refer to:. 

An astrolabe is an astronomical instrument dating to ancient times.  

An atmosphere is a layer of gases that envelop an astronomical object, held in place by the gravity of the object.  

Basalt is an aphanitic (fine-grained) extrusive igneous rock formed from the rapid cooling of low-viscosity lava rich in magnesium and iron. 

Biodiversity is the variability of life on Earth.  

A biome is a distinct geographical region with specific climate, vegetation, animal life, and an ecosystem.  

Biometrics are body measurements and calculations related to human characteristics and features.  

Boundary or Boundaries may refer to:. 

Bureaucracy is a system of organization where laws or regulatory authority are implemented by civil servants.  

In Western musical theory, a cadence is the end of a phrase in which the melody or harmony creates a sense of full or partial resolution. 

Calcification is the accumulation of calcium salts in a body tissue.  

In measurement technology and metrology, calibration is the comparison of measurement values delivered by a device under test with those of a calibration standard of known accuracy.  

The Cambrian is the first geological period of the Paleozoic Era and the Phanerozoic Eon.  

Capacitance is the ability of an object to store electric charge.  

A carcinogen is any agent that promotes the development of cancer.  

Causality is an influence by which one event, process, state, or subject contributes to the production of another event, process, state, or object. 

Cavitation in fluid mechanics and engineering normally is the phenomenon in which the static pressure of a liquid reduces to below the liquid's vapor pressure. 

Cellulose is an organic compound with the formula (C6H10O5)n, a polysaccharide consisting of a linear chain of several hundred to many thousands of β(1→4) linked D-glucose units.  

A centrifuge is a device that uses centrifugal force to subject a specimen to a specified constant force. 

A ceramic is any of the various hard, brittle, heat-resistant, and corrosion-resistant materials made by shaping and then firing an inorganic, nonmetallic material. 

Chlorophyll is any of several related green pigments found in cyanobacteria and in the chloroplasts of algae and plants.  

A chromosome is a package of DNA containing part or all of the genetic material of an organism.  

In cryptography, a cipher is an algorithm for performing encryption or decryption. 

An electronic circuit is composed of individual electronic components, such as resistors, transistors, capacitors, inductors and diodes. 

Cognitions are mental processes that deal with knowledge.  

Coherence is, in general, a state or situation in which all the parts or ideas fit together well so that they form a united whole. 

A colloid is a mixture in which one substance, consisting of microscopically dispersed insoluble particles, is suspended throughout another substance.  

Combustion, or burning, is a high-temperature exothermic redox chemical reaction between a fuel and an oxidant. 

In economics, a commodity is an economic good, usually a resource, that specifically has full or substantial fungibility. 

A press release is an official statement delivered to members of the news media for the purpose of providing new information. 

Conductor or conduction may refer to:. 

Conifers are a group of vascular plants and a subset of gymnosperms.  

Consensus usually refers to general agreement among a group of people or community.  

A constellation is an area on the celestial sphere in which a group of visible stars forms a perceived pattern or outline. 

Constraint may refer to:Constraint, a demarcation of geometrical characteristics between two or more entities or solid modeling bodies 

A dialect is a variety of language spoken by a particular group of people.  

Diffraction is the deviation of waves from straight-line propagation without any change in their energy due to an obstacle or through an aperture.  

Diffusion is the net movement of anything generally from a region of higher concentration to a region of lower concentration.  

Digestion is the breakdown of large insoluble food compounds into small water-soluble components. 

Dinosaurs are a diverse group of reptiles of the clade Dinosauria.  

Discourse is a generalization of the notion of a conversation to any form of communication.  

Displacement may refer to:. 

Diversity, diversify, or diverse may refer to:. 

Dopamine is a neuromodulatory molecule that plays several important roles in cells.  

A drought is a period of drier-than-normal conditions.  

Ductility is the ability of a material to sustain significant plastic deformation before fracture when undergoing tension. 

An ecosystem is a system formed by organisms in interaction with their environment.  

Eddy may refer to:Eddy (surname), surname used by descendants of a number of English, Irish and Scottish families 

Egalitarianism is a school of thought within political philosophy that builds on the concept of social equality. 

Elasticity often refers to:Elasticity (physics), continuum mechanics of bodies that deform reversibly under stress. 

An electrode is an electrical conductor used to make contact with a nonmetallic part of a circuit.  

In chemistry and manufacturing, electrolysis is a technique that uses direct electric current (DC) to drive an otherwise non-spontaneous biological and physical reaction.  

An elemental is a mythic supernatural being that is described in occult and alchemical works from around the time of the European Renaissance. 

The elevation of a geographic location is its height above or below a fixed reference point. 

Elimination may refer to:. 

An emulsion is a mixture of two or more liquids that are normally immiscible owing to liquid-liquid phase separation.  

The endocrine system is a messenger system in an organism comprising feedback loops of hormones that are released by internal glands directly into the circulatory system. 

An endoskeleton is a structural frame (skeleton) — usually composed of mineralized tissue — on the inside of an animal. 

Entropy is a scientific concept, most commonly associated with states of disorder, randomness, or uncertainty.  

An enzyme is a biological macromolecule, usually a protein, that acts as a biological catalyst. 

An epidemic is the rapid spread of disease to a large number of hosts in a given population within a short period of time.  

Epigenetics is the study of changes in gene expression that occur without altering the DNA sequence.  

Equilibrium may refer to:. 

Erosion is the action of surface processes that removes soil, rock, or dissolved material from one location on the Earth's crust. 

An estuary is a partially enclosed coastal body of brackish water where freshwater from rivers or streams meets and mixes with saltwater from the open sea.  

Ethnography is a branch of anthropology and the systematic study of individual cultures.  

Etiology is the study of causation or origination.  

Evolutionism is a term used to denote the theory of evolution.  

Excavation may refer to:Archaeological excavation 

Excitation, excite, exciting, or excitement may refer to:Excitation (magnetic), provided with an electrical generator or alternator 

An exoplanet or extrasolar planet is a planet outside of the Solar System.  

Extinction is the termination of a species via the death of its last member.  

A famine is a widespread scarcity of food caused by several possible factors. 

A fandom is a subculture composed of fans characterized by a feeling of camaraderie with others who share a common interest.  

Fertility in colloquial terms refers the ability to have offspring.  

Fiction is any creative work, chiefly any narrative work, portraying individuals, events, or places that are imaginary. 

Filtration is a physical separation process that separates solid matter and fluid from a mixture using a filter medium. 

Fission, a splitting of something into two or more parts, may refer to:. 

Flora is all the plant life present in a particular region or time. 

Flotation involves phenomena related to the relative buoyancy of objects. 

Flux describes any effect that appears to pass or travel through a surface or substance.  

A leaf is a principal appendage of the stem of a vascular plant, usually borne laterally above ground and specialized for photosynthesis.  

Forensic science, often confused with criminalistics, is the application of science principles and methods to support decision-making related to rules or law. 

In speech science and phonetics, a formant is the broad spectral maximum that results from an acoustic resonance of the human vocal tract.  

Formation may refer to:. 

Fracture is the appearance of a crack or complete separation of an object or material into two or more pieces under the action of stress.  

Friction is the force resisting the relative motion of solid surfaces, fluid layers, and material elements sliding or grinding against each other.  

Frost is a layer of ice on a solid surface, which forms from water vapor that deposits onto a freezing surface.  

In classical music, a fugue is a contrapuntal, polyphonic compositional technique in two or more voices. 

A fungus is any member of the group of eukaryotic organisms that includes microorganisms such as yeasts and molds. 
"""


# ══════════════════════════════════════════════════════════════════════════════
#  BPE-LITE SUBWORD TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """Byte-Pair Encoding lite: learns merge rules from a corpus, then
    tokenizes new text into subword units.  Falls back to character-level
    if no merges are known."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]

    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.trained = False

    # ── training ─────────────────────────────────────────────────────────

    def _word_freqs(self, text: str) -> Dict[str, int]:
        """Split text into words (space-separated), count frequencies."""
        words = re.findall(r'\S+', text.lower())
        freq: Dict[str, int] = Counter(words)
        # Represent each word as a tuple of characters + end-of-word marker
        return {tuple(w) + ("</w>",): c for w, c in freq.items()}

    @staticmethod
    def _pair_freqs(word_freqs: Dict[tuple, int]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: Tuple[str, str],
                    word_freqs: Dict[tuple, int]) -> Dict[tuple, int]:
        merged = pair[0] + pair[1]
        new_freqs: Dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            new_word: list = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def train(self, corpus: str, verbose: bool = False):
        """Learn BPE merge rules from *corpus*."""
        word_freqs = self._word_freqs(corpus)
        self.merges = []
        for _ in range(self.vocab_size - 256):  # 256 base chars
            pairs = self._pair_freqs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # type: ignore[arg-type]
            word_freqs = self._merge_pair(best, word_freqs)
            self.merges.append(best)

        # Build vocabulary
        self.vocab = {}
        idx = 0
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = idx
            idx += 1
        # Single characters
        for i in range(256):
            ch = chr(i)
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1
        # Merged tokens
        for a, b in self.merges:
            merged = a + b
            if merged not in self.vocab:
                self.vocab[merged] = idx
                idx += 1
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.trained = True
        if verbose:
            print(f"  BPE tokenizer: {len(self.vocab)} tokens, "
                  f"{len(self.merges)} merges learned.")

    # ── encoding / decoding ──────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Tokenize *text* into subword token ids."""
        if not self.trained:
            # Fallback: simple word-level ids
            return [hash(w) % (self.vocab_size or 2000) for w in text.lower().split()]
        words = re.findall(r'\S+', text.lower())
        ids: List[int] = []
        for word in words:
            symbols = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i:i + 2] = [a + b]
                    else:
                        i += 1
            for sym in symbols:
                ids.append(self.vocab.get(sym, self.vocab.get("<UNK>", 1)))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        tokens = [self.inverse_vocab.get(i, "<UNK>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING  (Sinusoidal, as in "Attention Is All You Need")
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding:
    """Deterministic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len
        self.table = self._build_table()

    def _build_table(self) -> List[List[float]]:
        table = []
        for pos in range(self.max_len):
            row = []
            for i in range(self.d_model):
                angle = pos / (10000 ** ((2 * (i // 2)) / self.d_model))
                if i % 2 == 0:
                    row.append(math.sin(angle))
                else:
                    row.append(math.cos(angle))
            table.append(row)
        return table

    def encode(self, seq_len: int) -> List[List[float]]:
        """Return positional vectors for positions 0..seq_len-1."""
        return self.table[:seq_len]


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING LAYER
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingLayer:
    """Trainable token embedding + positional encoding."""

    def __init__(self, vocab_size: int, d_model: int = 64):
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Xavier-uniform initialization
        scale = math.sqrt(6.0 / (vocab_size + d_model))
        self.weight: List[List[float]] = [
            [random.uniform(-scale, scale) for _ in range(d_model)]
            for _ in range(vocab_size)
        ]
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """Return (seq_len x d_model) matrix: embedding + positional."""
        seq_len = len(token_ids)
        pos = self.pos_enc.encode(seq_len)
        out = []
        for i, tid in enumerate(token_ids):
            if tid < self.vocab_size:
                vec = [self.weight[tid][j] + pos[i][j] for j in range(self.d_model)]
            else:
                vec = list(pos[i])  # fallback for OOV
            out.append(vec)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm:
    """Per-vector layer normalization with learnable scale and shift."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma: List[float] = [1.0] * d_model
        self.beta: List[float] = [0.0] * d_model

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        out = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            out.append([
                self.gamma[j] * (row[j] - mean) / math.sqrt(var + self.eps) + self.beta[j]
                for j in range(len(row))
            ])
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-HEAD SELF-ATTENTION  (Pure Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention:
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Initialize Q, K, V projection matrices and output projection
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = self._rand_matrix(d_model, d_model, scale)
        self.W_k = self._rand_matrix(d_model, d_model, scale)
        self.W_v = self._rand_matrix(d_model, d_model, scale)
        self.W_o = self._rand_matrix(d_model, d_model, scale)

    @staticmethod
    def _rand_matrix(rows: int, cols: int, scale: float) -> List[List[float]]:
        return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Matrix multiply  (m x n) @ (n x p) -> (m x p)."""
        m, n, p = len(a), len(a[0]), len(b[0])
        out = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                aik = a[i][k]
                for j in range(p):
                    out[i][j] += aik * b[k][j]
        return out

    @staticmethod
    def _softmax(vec: List[float]) -> List[float]:
        m = max(vec)
        exps = [math.exp(v - m) for v in vec]
        s = sum(exps)
        return [e / s for e in exps]

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """x: (seq_len, d_model) -> (seq_len, d_model)."""
        seq_len = len(x)
        # Project to Q, K, V
        Q = self._matmul(x, self.W_q)  # (seq_len, d_model)
        K = self._matmul(x, self.W_k)
        V = self._matmul(x, self.W_v)

        # Split into heads and compute attention per head
        heads_out: List[List[List[float]]] = []
        for h in range(self.n_heads):
            offset = h * self.d_k
            q_h = [row[offset:offset + self.d_k] for row in Q]
            k_h = [row[offset:offset + self.d_k] for row in K]
            v_h = [row[offset:offset + self.d_k] for row in V]

            # Scaled dot-product attention:  scores = Q_h @ K_h^T / sqrt(d_k)
            scale = math.sqrt(self.d_k)
            attn_scores: List[List[float]] = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    dot = sum(q_h[i][d] * k_h[j][d] for d in range(self.d_k))
                    row.append(dot / scale)
                attn_scores.append(row)

            # Softmax over keys (causal mask: only attend to j <= i)
            attn_weights: List[List[float]] = []
            for i in range(seq_len):
                masked = [attn_scores[i][j] if j <= i else -1e9 for j in range(seq_len)]
                attn_weights.append(self._softmax(masked))

            # Weighted sum of values
            head_out: List[List[float]] = []
            for i in range(seq_len):
                vec = [0.0] * self.d_k
                for j in range(seq_len):
                    w = attn_weights[i][j]
                    for d in range(self.d_k):
                        vec[d] += w * v_h[j][d]
                head_out.append(vec)
            heads_out.append(head_out)

        # Concatenate heads
        concat: List[List[float]] = []
        for i in range(seq_len):
            row: List[float] = []
            for h in range(self.n_heads):
                row.extend(heads_out[h][i])
            concat.append(row)

        # Output projection
        return self._matmul(concat, self.W_o)


# ══════════════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK  (2-layer MLP with GELU)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward:
    """Position-wise feed-forward network:  d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int = 64, d_ff: int = 256):
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))
        self.W1 = [[random.uniform(-scale1, scale1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.uniform(-scale2, scale2) for _ in range(d_model)] for _ in range(d_ff)]
        self.b2 = [0.0] * d_model

    @staticmethod
    def _gelu(x: float) -> float:
        """Gaussian Error Linear Unit approximation."""
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        out = []
        for row in x:
            # Hidden layer
            hidden = [
                self._gelu(sum(row[k] * self.W1[k][j] for k in range(len(row))) + self.b1[j])
                for j in range(len(self.b1))
            ]
            # Output layer
            proj = [
                sum(hidden[k] * self.W2[k][j] for k in range(len(hidden))) + self.b2[j]
                for j in range(len(self.b2))
            ]
            out.append(proj)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK  (Attention + FFN + Residual + LayerNorm)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock:
    """Pre-norm transformer block:  LN -> Attn -> Residual -> LN -> FFN -> Residual."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        # Self-attention with residual
        normed = self.ln1.forward(x)
        attn_out = self.attn.forward(normed)
        x = [  # residual add
            [x[i][j] + attn_out[i][j] for j in range(len(x[0]))]
            for i in range(len(x))
        ]
        # Feed-forward with residual
        normed2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(normed2)
        x = [
            [x[i][j] + ffn_out[i][j] for j in range(len(x[0]))]
            for i in range(len(x))
        ]
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL ENGINE  v5  (15+ tools)
# ══════════════════════════════════════════════════════════════════════════════

class ToolEngine:
    """Route user queries to the appropriate tool and return structured results."""

    _SAFE_MATH_NS = {
        "__builtins__": None,
        "math": math, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "factorial": math.factorial, "abs": abs, "round": round,
        "pow": pow, "min": min, "max": max, "sum": sum,
        "gcd": math.gcd, "ceil": math.ceil, "floor": math.floor,
        "exp": math.exp, "radians": math.radians, "degrees": math.degrees,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    }

    # ── 1. BARE MATH ─────────────────────────────────────────────────────

    _SAFE_NAMES = {
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log2', 'log10',
        'factorial', 'abs', 'exp', 'ceil', 'floor', 'gcd',
        'min', 'max', 'sum', 'pow', 'round', 'pi', 'e',
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
        'radians', 'degrees', 'math',
    }

    @classmethod
    def _safe_eval(cls, expr: str):
        try:
            # Allow digits, operators, parens, and whitelisted function names
            cleaned = re.sub(r'[a-zA-Z_]+', lambda m: m.group() if m.group() in cls._SAFE_NAMES else '', expr)
            if not cleaned.strip() or cleaned != expr:
                # If anything was removed, it wasn't a safe expression
                # But allow through if only safe names were found
                if cleaned.strip() != expr.strip():
                    # Check that all word tokens are safe
                    words = re.findall(r'[a-zA-Z_]+', expr)
                    if not all(w in cls._SAFE_NAMES for w in words):
                        return None
            result = eval(expr, cls._SAFE_MATH_NS)
            if isinstance(result, float):
                return int(result) if result == int(result) else round(result, 10)
            return result
        except Exception:
            return None

    # ── 2. BASE64 ────────────────────────────────────────────────────────

    @classmethod
    def _tool_base64(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:b64|base64)\s+(enc|encode|dec|decode)\s+(.+)', q, re.I)
        if not m:
            return None
        mode, text = m.groups()
        try:
            if mode.lower().startswith('enc'):
                return f"Base64 encode: {base64.b64encode(text.encode()).decode()}"
            else:
                return f"Base64 decode: {base64.b64decode(text.encode()).decode('utf-8', errors='ignore')}"
        except Exception:
            return "Error: Could not process base64"

    # ── 3. HASH ──────────────────────────────────────────────────────────

    @classmethod
    def _tool_hash(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:hash|md5|sha1|sha256|sha512)\s+(.+)', q, re.I)
        if not m:
            return None
        full = m.group(0).lower()
        for algo in ['sha512', 'sha256', 'sha1', 'md5']:
            if full.startswith(algo):
                text = q[len(algo):].strip()
                h = hashlib.new(algo)
                h.update(text.encode())
                return f"Hash ({algo}): {h.hexdigest()}"
        return "Error: Unknown hash algorithm. Use md5, sha1, sha256, or sha512."

    # ── 4. FRACTIONS ─────────────────────────────────────────────────────

    @classmethod
    def _tool_fraction(cls, q: str) -> Optional[str]:
        if '/' not in q:
            return None
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', q):
            return None
        try:
            safe_ns = {"Fraction": Fraction, "__builtins__": None}
            res = eval(q, safe_ns)
            if isinstance(res, Fraction):
                return f"Fraction: {res}  (decimal: {float(res):.6f})"
        except Exception:
            pass
        return None

    # ── 5. COMPLEX MATH ──────────────────────────────────────────────────

    @classmethod
    def _tool_complex(cls, q: str) -> Optional[str]:
        if 'j' not in q.lower() and not re.search(r'sqrt\s*\(\s*-', q):
            return None
        try:
            expr = q.strip()
            if 'j' in expr:
                # Only replace standalone 'j' (not preceded by a digit)
                # so "4j" stays "4j" (Python already understands it) but
                # a bare "j" becomes "1j"
                expr = re.sub(r'(?<!\d)j', '1j', expr)
            ns = {"__builtins__": None, "cmath": cmath, "math": math,
                  "sqrt": cmath.sqrt, "sin": cmath.sin, "cos": cmath.cos,
                  "tan": cmath.tan, "log": cmath.log, "exp": cmath.exp,
                  "pi": math.pi, "e": math.e}
            res = eval(expr, ns)
            return f"Complex: {res}"
        except Exception:
            return None

    # ── 6. UNIT CONVERTER ────────────────────────────────────────────────

    _UNITS: Dict[str, Dict[str, float]] = {
        # Length (to meters)
        "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
                   "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
        # Mass (to kg)
        "mass": {"mg": 1e-6, "g": 0.001, "kg": 1, "lb": 0.453592, "oz": 0.0283495, "ton": 1000},
        # Temperature (special handling)
        "temperature": {"c": 0, "f": 0, "k": 0},
        # Volume (to liters)
        "volume": {"ml": 0.001, "l": 1, "gal": 3.78541, "qt": 0.946353,
                   "pt": 0.473176, "cup": 0.236588, "floz": 0.0295735},
        # Speed (to m/s)
        "speed": {"ms": 1, "kmh": 0.277778, "mph": 0.44704, "knot": 0.514444, "fps": 0.3048},
        # Data (to bytes)
        "data": {"b": 1, "kb": 1024, "mb": 1048576, "gb": 1073741824,
                 "tb": 1099511627776, "pb": 1125899906842624},
    }

    @classmethod
    def _tool_unit(cls, q: str) -> Optional[str]:
        m = re.match(
            r'^(?:convert|unit)\s+([0-9.]+)\s+(\w+)\s+(?:to|in|as)\s+(\w+)', q, re.I
        )
        if not m:
            return None
        value, from_u, to_u = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
        for cat, units in cls._UNITS.items():
            if from_u in units and to_u in units:
                if cat == "temperature":
                    result = cls._convert_temp(value, from_u, to_u)
                else:
                    base = value * units[from_u]
                    result = base / units[to_u]
                return f"{value} {from_u} = {round(result, 6)} {to_u}"
        return "Error: Unsupported unit conversion. Check unit names."

    @staticmethod
    def _convert_temp(val: float, fr: str, to: str) -> float:
        # Convert to Celsius first
        if fr == "c":
            c = val
        elif fr == "f":
            c = (val - 32) * 5 / 9
        else:  # K
            c = val - 273.15
        # Convert from Celsius to target
        if to == "c":
            return c
        elif to == "f":
            return c * 9 / 5 + 32
        else:  # K
            return c + 273.15

    # ── 7. DATE / TIME ───────────────────────────────────────────────────

    @classmethod
    def _tool_date(cls, q: str) -> Optional[str]:
        q_low = q.lower()
        if q_low in ['date', 'time', 'now', 'today', 'datetime']:
            now = datetime.datetime.now()
            return f"Current: {now.strftime('%Y-%m-%d %H:%M:%S %A')}"
        # "days from 2024-01-01 to 2024-12-31"
        m = re.match(r'^days\s+(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            d1 = datetime.date.fromisoformat(m.group(1))
            d2 = datetime.date.fromisoformat(m.group(2))
            delta = abs((d2 - d1).days)
            return f"Days between: {delta} days"
        # "add 30 days to 2024-01-01"
        m = re.match(r'^add\s+(\d+)\s+days?\s+(?:to\s+)?(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            ndays = int(m.group(1))
            d = datetime.date.fromisoformat(m.group(2))
            result = d + datetime.timedelta(days=ndays)
            return f"Result: {result.strftime('%Y-%m-%d %A')}"
        # "day of week 2024-06-15"
        m = re.match(r'^day\s+(?:of\s+week\s+)?(\d{4}-\d{2}-\d{2})', q, re.I)
        if m:
            d = datetime.date.fromisoformat(m.group(1))
            return f"{d.strftime('%Y-%m-%d')} is a {d.strftime('%A')}"
        return None

    # ── 8. COLOR CONVERTER ───────────────────────────────────────────────

    @classmethod
    def _tool_color(cls, q: str) -> Optional[str]:
        # "color #ff8800" or "color rgb 255 128 0" or "color hsl 30 100 50"
        m = re.match(r'^color\s+#([0-9a-f]{6})', q, re.I)
        if m:
            h = m.group(1)
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            hsl = cls._rgb_to_hsl(r, g, b)
            return f"HEX #{h} -> RGB({r}, {g}, {b}) -> HSL({hsl[0]}, {hsl[1]}%, {hsl[2]}%)"
        m = re.match(r'^color\s+rgb\s+(\d+)\s+(\d+)\s+(\d+)', q, re.I)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hx = f"#{r:02x}{g:02x}{b:02x}"
            hsl = cls._rgb_to_hsl(r, g, b)
            return f"RGB({r}, {g}, {b}) -> HEX {hx} -> HSL({hsl[0]}, {hsl[1]}%, {hsl[2]}%)"
        m = re.match(r'^color\s+hsl\s+(\d+)\s+(\d+)\s+(\d+)', q, re.I)
        if m:
            h, s, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rgb = cls._hsl_to_rgb(h, s, l)
            hx = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            return f"HSL({h}, {s}%, {l}%) -> RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) -> HEX {hx}"
        return None

    @staticmethod
    def _rgb_to_hsl(r: int, g: int, b: int) -> Tuple[int, int, int]:
        r1, g1, b1 = r / 255, g / 255, b / 255
        mx, mn = max(r1, g1, b1), min(r1, g1, b1)
        l = (mx + mn) / 2
        if mx == mn:
            h = s = 0
        else:
            d = mx - mn
            s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == r1:
                h = ((g1 - b1) / d + (6 if g1 < b1 else 0)) / 6
            elif mx == g1:
                h = ((b1 - r1) / d + 2) / 6
            else:
                h = ((r1 - g1) / d + 4) / 6
        return round(h * 360), round(s * 100), round(l * 100)

    @staticmethod
    def _hsl_to_rgb(h: int, s: int, l: int) -> Tuple[int, int, int]:
        h1, s1, l1 = h / 360, s / 100, l / 100
        if s1 == 0:
            v = round(l1 * 255)
            return v, v, v

        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        q = l1 * (1 + s1) if l1 < 0.5 else l1 + s1 - l1 * s1
        p = 2 * l1 - q
        return (round(hue2rgb(p, q, h1 + 1/3) * 255),
                round(hue2rgb(p, q, h1) * 255),
                round(hue2rgb(p, q, h1 - 1/3) * 255))

    # ── 9. ROMAN NUMERALS ────────────────────────────────────────────────

    @classmethod
    def _tool_roman(cls, q: str) -> Optional[str]:
        # "roman 42" or "roman XLII"
        m = re.match(r'^roman\s+([IVXLCDM]+)$', q, re.I)
        if m:
            try:
                val = cls._roman_to_int(m.group(1).upper())
                return f"Roman {m.group(1).upper()} = {val}"
            except ValueError:
                return "Error: Invalid Roman numeral."
        m = re.match(r'^roman\s+(\d+)$', q, re.I)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 3999:
                return f"{val} = {cls._int_to_roman(val)}"
            return "Error: Number must be between 1 and 3999."
        return None

    @staticmethod
    def _roman_to_int(s: str) -> int:
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev = 0
        for ch in reversed(s):
            v = vals.get(ch, 0)
            if v < prev:
                total -= v
            else:
                total += v
            prev = v
        if total == 0:
            raise ValueError
        return total

    @staticmethod
    def _int_to_roman(n: int) -> str:
        pairs = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                 (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                 (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
        result = ''
        for val, sym in pairs:
            while n >= val:
                result += sym
                n -= val
        return result

    # ── 10. STATISTICS ───────────────────────────────────────────────────

    @classmethod
    def _tool_stats(cls, q: str) -> Optional[str]:
        m = re.match(r'^stats?\s+([\d.,\s]+)$', q, re.I)
        if not m:
            return None
        try:
            nums = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
        except ValueError:
            return "Error: Could not parse numbers."
        if len(nums) < 2:
            return "Error: Need at least 2 numbers."
        n = len(nums)
        mean = sum(nums) / n
        sorted_n = sorted(nums)
        median = (sorted_n[n // 2] if n % 2 == 1
                  else (sorted_n[n // 2 - 1] + sorted_n[n // 2]) / 2)
        freq = Counter(nums)
        max_freq = max(freq.values())
        modes = [k for k, v in freq.items() if v == max_freq]
        mode_str = ", ".join(str(m) for m in modes) if max_freq > 1 else "No mode"
        variance = sum((x - mean) ** 2 for x in nums) / (n - 1)
        stdev = math.sqrt(variance)
        rng = max(nums) - min(nums)
        return (f"Count: {n}  |  Mean: {round(mean, 4)}  |  Median: {round(median, 4)}\n"
                f"Mode: {mode_str}  |  Stdev: {round(stdev, 4)}  |  Variance: {round(variance, 4)}\n"
                f"Range: {round(rng, 4)}  |  Min: {min(nums)}  |  Max: {max(nums)}")

    # ── 11. COMBINATORICS ────────────────────────────────────────────────

    @classmethod
    def _tool_combinatorics(cls, q: str) -> Optional[str]:
        # "ncr 10 3" or "npr 5 2" or "factorial 6"
        m = re.match(r'^(?:ncr|C)\s+(\d+)\s+(\d+)$', q, re.I)
        if m:
            n, r = int(m.group(1)), int(m.group(2))
            if r > n:
                return "Error: r cannot be greater than n."
            return f"C({n},{r}) = {math.comb(n, r)}"
        m = re.match(r'^(?:npr|P)\s+(\d+)\s+(\d+)$', q, re.I)
        if m:
            n, r = int(m.group(1)), int(m.group(2))
            if r > n:
                return "Error: r cannot be greater than n."
            return f"P({n},{r}) = {math.perm(n, r)}"
        m = re.match(r'^(?:factorial|fact)\s+(\d+)$', q, re.I)
        if m:
            n = int(m.group(1))
            if n > 170:
                return "Error: Number too large (max 170)."
            return f"{n}! = {math.factorial(n)}"
        return None

    # ── 12. NUMBER-BASE CONVERTER ────────────────────────────────────────

    @classmethod
    def _tool_baseconv(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:base|conv)\s+([0-9a-fA-F]+)\s+(?:from\s+)?(bin|oct|dec|hex)\s+(?:to\s+)?(bin|oct|dec|hex)', q, re.I)
        if not m:
            # Shortcut: "hex 255" or "bin 10" or "oct 64"
            m2 = re.match(r'^(hex|bin|oct)\s+(\d+)$', q, re.I)
            if m2:
                target, val = m2.group(1).lower(), m2.group(2)
                dec_val = int(val, 10)
                results = {"hex": hex, "bin": bin, "oct": oct}
                return f"{val} (dec) = {results[target](dec_val)}"
            return None
        val_str, from_base, to_base = m.group(1), m.group(2).lower(), m.group(3).lower()
        bases = {"bin": 2, "oct": 8, "dec": 10, "hex": 16}
        try:
            dec_val = int(val_str, bases[from_base])
        except ValueError:
            return "Error: Invalid number for the given base."
        if to_base == "dec":
            return f"{val_str} ({from_base}) = {dec_val} (dec)"
        elif to_base == "hex":
            return f"{val_str} ({from_base}) = {hex(dec_val)} (hex)"
        elif to_base == "oct":
            return f"{val_str} ({from_base}) = {oct(dec_val)} (oct)"
        else:
            return f"{val_str} ({from_base}) = {bin(dec_val)} (bin)"

    # ── 13. GEOMETRY ─────────────────────────────────────────────────────

    @classmethod
    def _tool_geometry(cls, q: str) -> Optional[str]:
        q_low = q.lower()
        # Circle
        m = re.match(r'^(?:area|circumference)\s+(?:of\s+)?circle\s+r\s*=?\s*([0-9.]+)', q_low)
        if m:
            r = float(m.group(1))
            area = math.pi * r ** 2
            circ = 2 * math.pi * r
            return f"Circle (r={r}): Area = {round(area, 4)}, Circumference = {round(circ, 4)}"
        # Rectangle
        m = re.match(r'^(?:area|perimeter)\s+(?:of\s+)?rect(?:angle)?\s+([0-9.]+)\s*[x,]\s*([0-9.]+)', q_low)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            return f"Rectangle ({w}x{h}): Area = {w*h}, Perimeter = {2*(w+h)}"
        # Triangle
        m = re.match(r'^(?:area)\s+(?:of\s+)?tri(?:angle)?\s+(?:base\s*=?\s*([0-9.]+)\s+height\s*=?\s*([0-9.]+))', q_low)
        if m:
            b, h = float(m.group(1)), float(m.group(2))
            return f"Triangle (b={b}, h={h}): Area = {round(0.5*b*h, 4)}"
        # Sphere
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?sphere\s+r\s*=?\s*([0-9.]+)', q_low)
        if m:
            r = float(m.group(1))
            vol = 4/3 * math.pi * r**3
            sa = 4 * math.pi * r**2
            return f"Sphere (r={r}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Cylinder
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?cylinder\s+r\s*=?\s*([0-9.]+)\s+h\s*=?\s*([0-9.]+)', q_low)
        if m:
            r, h = float(m.group(1)), float(m.group(2))
            vol = math.pi * r**2 * h
            sa = 2 * math.pi * r * (r + h)
            return f"Cylinder (r={r}, h={h}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Cone
        m = re.match(r'^(?:area|volume)\s+(?:of\s+)?cone\s+r\s*=?\s*([0-9.]+)\s+h\s*=?\s*([0-9.]+)', q_low)
        if m:
            r, h = float(m.group(1)), float(m.group(2))
            vol = math.pi * r**2 * h / 3
            sl = math.sqrt(r**2 + h**2)
            sa = math.pi * r * (r + sl)
            return f"Cone (r={r}, h={h}): Volume = {round(vol, 4)}, Surface Area = {round(sa, 4)}"
        # Pythagorean
        m = re.match(r'^pyth(?:agorean)?\s+([0-9.]+)\s+([0-9.]+)', q_low)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            c = math.sqrt(a**2 + b**2)
            return f"Hypotenuse: {round(c, 4)}  (a={a}, b={b})"
        return None

    # ── 14. FINANCIAL CALCULATOR ─────────────────────────────────────────

    @classmethod
    def _tool_finance(cls, q: str) -> Optional[str]:
        # Compound interest: "compound 1000 5 10" (principal, rate%, years)
        m = re.match(r'^compound\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
            a = p * (1 + r/100) ** t
            interest = a - p
            return (f"Principal: {p}  |  Rate: {r}%  |  Years: {t}\n"
                    f"Final Amount: {round(a, 2)}  |  Interest Earned: {round(interest, 2)}")
        # Loan payment: "loan 100000 5 30" (principal, rate%, years)
        m = re.match(r'^loan\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            p, r, t = float(m.group(1)), float(m.group(2)), float(m.group(3))
            monthly_rate = r / 100 / 12
            n_months = t * 12
            if monthly_rate == 0:
                payment = p / n_months
            else:
                payment = p * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
            total_paid = payment * n_months
            return (f"Loan: {p}  |  Rate: {r}%  |  Term: {t} years\n"
                    f"Monthly Payment: {round(payment, 2)}  |  Total Paid: {round(total_paid, 2)}  |  "
                    f"Total Interest: {round(total_paid - p, 2)}")
        # ROI: "roi 1000 1500" (investment, return)
        m = re.match(r'^roi\s+([0-9.]+)\s+([0-9.]+)', q, re.I)
        if m:
            cost, gain = float(m.group(1)), float(m.group(2))
            roi = (gain - cost) / cost * 100
            profit = gain - cost
            return f"Investment: {cost}  |  Return: {gain}  |  Profit: {round(profit, 2)}  |  ROI: {round(roi, 2)}%"
        return None

    # ── 15. GCD / LCM ────────────────────────────────────────────────────

    @classmethod
    def _tool_gcd_lcm(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:gcd|lcm)\s+(\d+)\s+(\d+)', q, re.I)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        g = math.gcd(a, b)
        l = abs(a * b) // g
        cmd = m.group(0).split()[0].lower()
        if cmd == "gcd":
            return f"GCD({a}, {b}) = {g}"
        return f"LCM({a}, {b}) = {l}"

    # ── 16. LINEAR EQUATION SOLVER ───────────────────────────────────────

    @classmethod
    def _tool_equation(cls, q: str) -> Optional[str]:
        # "solve 2x + 3 = 11"
        m = re.match(r'^solve\s+([0-9.]*)\s*x\s*([+\-])\s*([0-9.]+)\s*=\s*([0-9.]+)', q, re.I)
        if m:
            a = float(m.group(1)) if m.group(1) else 1.0
            sign = 1 if m.group(2) == '+' else -1
            b = float(m.group(3))
            c = float(m.group(4))
            x = (c - sign * b) / a
            return f"{a}x {'+' if sign == 1 else '-'} {b} = {c}  =>  x = {round(x, 6)}"
        # "solve 3x - 7 = 20"
        m = re.match(r'^solve\s+([0-9.]*)\s*x\s*([+\-])\s*([0-9.]+)\s*=\s*([0-9.]+)', q, re.I)
        if m:
            a = float(m.group(1)) if m.group(1) else 1.0
            sign = 1 if m.group(2) == '+' else -1
            b = float(m.group(3))
            c = float(m.group(4))
            x = (c - sign * b) / a
            return f"x = {round(x, 6)}"
        return None

    # ── 17. TEXT ANALYSIS ────────────────────────────────────────────────

    @classmethod
    def _tool_text_analysis(cls, q: str) -> Optional[str]:
        m = re.match(r'^(?:analyze|textstats?|wordcount)\s+(.+)$', q, re.I)
        if not m:
            return None
        text = m.group(1)
        words = text.split()
        chars = len(text)
        chars_no_space = len(text.replace(" ", ""))
        sentences = len(re.findall(r'[.!?]+', text)) or 1
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        # Flesch-Kincaid approximation
        syllables = sum(cls._count_syllables(w) for w in words)
        if len(words) > 0:
            fk = (0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59)
        else:
            fk = 0
        return (f"Words: {len(words)}  |  Characters: {chars}  |  Characters (no spaces): {chars_no_space}\n"
                f"Sentences: {sentences}  |  Avg word length: {round(avg_word_len, 1)}  |  "
                f"Syllables: {syllables}\n"
                f"Readability (Flesch-Kincaid grade): {round(max(0, fk), 1)}")

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    # ── 18. PROBABILITY CALCULATOR ───────────────────────────────────────

    @classmethod
    def _tool_probability(cls, q: str) -> Optional[str]:
        # Binomial: "binomial 10 0.5 3" (n, p, k) = P(X=k)
        m = re.match(r'^binomial\s+(\d+)\s+([0-9.]+)\s+(\d+)', q, re.I)
        if m:
            n, p, k = int(m.group(1)), float(m.group(2)), int(m.group(3))
            if not (0 <= p <= 1):
                return "Error: Probability p must be between 0 and 1."
            prob = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
            mean = n * p
            variance = n * p * (1 - p)
            return (f"Binomial(n={n}, p={p}, k={k}): P(X={k}) = {round(prob, 6)}\n"
                    f"Mean: {round(mean, 4)}  |  Variance: {round(variance, 4)}  |  Stdev: {round(math.sqrt(variance), 4)}")
        # Poisson: "poisson 5 3" (lambda, k) = P(X=k)
        m = re.match(r'^poisson\s+([0-9.]+)\s+(\d+)', q, re.I)
        if m:
            lam, k = float(m.group(1)), int(m.group(2))
            if lam <= 0:
                return "Error: Lambda must be positive."
            prob = (lam ** k) * math.exp(-lam) / math.factorial(k)
            return (f"Poisson(lambda={lam}, k={k}): P(X={k}) = {round(prob, 6)}\n"
                    f"Mean: {lam}  |  Variance: {lam}  |  Stdev: {round(math.sqrt(lam), 4)}")
        return None

    # ── MASTER ROUTER ────────────────────────────────────────────────────

    @classmethod
    def route(cls, query: str) -> Optional[str]:
        """Try every tool in order; return the first non-None result, or None."""
        q = query.strip()

        # 1. Bare math expression  (pure numbers/operators OR safe function calls)
        math_func_pattern = r'^(sqrt|sin|cos|tan|log2?|log10|factorial|abs|exp|ceil|floor|gcd|min|max|pow|round|asin|acos|atan|sinh|cosh|tanh|radians|degrees)\s*[\(\d]'
        if re.match(r'^[\d\s\+\-\*\/\%\.\(\)eE]+$', q) or re.match(math_func_pattern, q, re.I):
            result = cls._safe_eval(q)
            if result is not None:
                return f"Answer: {result}"

        # 2-18. Named tools
        for tool_fn in [
            cls._tool_base64,
            cls._tool_hash,
            cls._tool_fraction,
            cls._tool_complex,
            cls._tool_unit,
            cls._tool_date,
            cls._tool_color,
            cls._tool_roman,
            cls._tool_stats,
            cls._tool_combinatorics,
            cls._tool_baseconv,
            cls._tool_geometry,
            cls._tool_finance,
            cls._tool_gcd_lcm,
            cls._tool_equation,
            cls._tool_text_analysis,
            cls._tool_probability,
        ]:
            result = tool_fn(q)
            if result is not None:
                return result

        # Help
        if q.lower() in ['help', '/help', '!tools', 'tools']:
            return (
                "MLLM-5-BASE Tools:\n"
                "  Math: 2+2, sqrt(16), sin(0.5)\n"
                "  Base64: base64 encode hello / base64 decode SGVsbG8=\n"
                "  Hash: sha256 password / md5 hello\n"
                "  Fractions: 1/2 + 3/4\n"
                "  Complex: 3+4j / sqrt(-1)\n"
                "  Units: convert 100 km to mi / convert 72 kg to lb\n"
                "  Date: now / days from 2024-01-01 to 2024-12-31\n"
                "  Color: color #ff8800 / color rgb 255 128 0\n"
                "  Roman: roman 42 / roman XLII\n"
                "  Stats: stats 10,20,30,40,50\n"
                "  Combinatorics: ncr 10 3 / npr 5 2 / factorial 6\n"
                "  Base conv: hex 255 / base FF from hex to dec\n"
                "  Geometry: area circle r=5 / volume sphere r=3\n"
                "  Finance: compound 1000 5 10 / loan 100000 5 30\n"
                "  GCD/LCM: gcd 12 8 / lcm 4 6\n"
                "  Equation: solve 2x + 3 = 11\n"
                "  Text: analyze The quick brown fox\n"
                "  Probability: binomial 10 0.5 3 / poisson 5 3"
            )

        return None  # No tool matched


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT MEMORY  (Word-overlap retrieval from user queries only)
# ══════════════════════════════════════════════════════════════════════════════

class ContextMemory:
    """Stores conversation turns and retrieves the most relevant *user queries*
    to enrich generation context.  Model outputs are stored for display but are
    NOT injected into the generation context — this prevents the degenerate
    feedback loop where bad model output poisons future generation."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []  # {"user": ..., "model": ...}

    def add(self, user_text: str, model_text: str):
        self.turns.append({"user": user_text, "model": model_text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def retrieve_user_context(self, query: str, top_k: int = 2) -> str:
        """Return a string of the most relevant past *user queries* to prepend
        to the current context.  Model responses are excluded to avoid
        contaminating generation with low-quality output."""
        if not self.turns:
            return ""
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        if not q_words:
            return " ".join(t["user"] for t in self.turns[-top_k:])
        scored = []
        for turn in self.turns:
            turn_words = set(re.findall(r'\b\w+\b', turn["user"].lower()))
            overlap = len(q_words & turn_words)
            scored.append((overlap, turn["user"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(text for _, text in scored[:top_k])

    def get_recent(self, n: int = 3) -> List[Dict[str, str]]:
        return self.turns[-n:]

    def clear(self):
        self.turns = []


# ══════════════════════════════════════════════════════════════════════════════
#  N-GRAM MODEL  —  Fast Interpolated Backoff + Direct Lookup
# ══════════════════════════════════════════════════════════════════════════════

class SmoothedNGramModel:
    """N-gram language model with interpolated backoff.

    Key design choices for SPEED and ACCURACY on small corpora:
    • predict_next() only scores words that were ACTUALLY SEEN after the
      matched context — not the entire vocabulary.  This is O(k) where k
      is the number of distinct successors, not O(|V|).
    • Backoff is simple: try the longest matching n-gram first; if not
      found, chop one token from the left and try again.
    • When a context IS found, its successors are weighted by count
      (with optional temperature scaling).  No expensive smoothing math.
    • Fall back to unigram frequencies when no context matches.
    """

    def __init__(self, max_n: int = 20):
        self.max_n = max_n
        # n -> {context_tuple: Counter(next_token)}
        self.ngram_contexts: Dict[int, Dict[tuple, Counter]] = {}
        self.unigram_count = 0
        self.vocab: set = set()
        self.token_freq: Counter = Counter()
        self._trained = False

    def train(self, tokens: List[str]):
        """Build n-gram tables from the token list."""
        self.vocab = set(tokens)
        self.token_freq = Counter(tokens)
        self.unigram_count = len(tokens)

        for n in range(1, self.max_n + 1):
            ctx: Dict[tuple, Counter] = defaultdict(Counter)
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i + n])
                next_tok = tokens[i + n]
                ctx[ngram][next_tok] += 1
            self.ngram_contexts[n] = dict(ctx)
        self._trained = True

    def _get_successors(self, context_tokens: List[str]) -> Optional[Counter]:
        """Walk from longest n-gram down to unigram, return the first
        non-empty successor Counter, or None if nothing matches."""
        for n in range(min(len(context_tokens), self.max_n), 0, -1):
            ngram = tuple(context_tokens[-n:])
            ctx_dict = self.ngram_contexts.get(n, {}).get(ngram)
            if ctx_dict and len(ctx_dict) > 0:
                return ctx_dict
        return None

    def predict_next(self, context_tokens: List[str],
                     temperature: float = 0.8) -> Tuple[str, float]:
        """Return (predicted_word, confidence).

        Only scores words that were seen after the matched context —
        typically a small handful, NOT the whole vocabulary.  This is
        the single biggest speed win over the previous version.
        """
        if not self._trained or not self.vocab:
            return ("<UNK>", 0.0)

        # Fast path: direct successor lookup
        successors = self._get_successors(context_tokens)

        if successors:
            words = list(successors.keys())
            counts = list(successors.values())
            total = sum(counts)

            # Convert to probabilities
            probs = [c / total for c in counts]

            # Apply temperature: lower = sharper (more deterministic)
            if temperature != 1.0 and temperature > 0:
                probs = [p ** (1.0 / temperature) for p in probs]
                psum = sum(probs)
                probs = [p / psum for p in probs]

            # Weighted random choice
            chosen = random.choices(words, weights=probs, k=1)[0]
            confidence = probs[words.index(chosen)]
            return chosen, confidence

        # Fallback: sample from unigram distribution (weighted by frequency)
        words = list(self.token_freq.keys())
        counts = list(self.token_freq.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        if temperature != 1.0 and temperature > 0:
            probs = [p ** (1.0 / temperature) for p in probs]
            psum = sum(probs)
            probs = [p / psum for p in probs]

        chosen = random.choices(words, weights=probs, k=1)[0]
        confidence = probs[words.index(chosen)]
        return chosen, confidence


# ══════════════════════════════════════════════════════════════════════════════
#  ONLINE LEARNER  (Incremental n-gram updates — USER INPUT ONLY)
# ══════════════════════════════════════════════════════════════════════════════

class OnlineLearner:
    """Incrementally updates n-gram counts as new text is observed.
    CRITICAL: only learn from high-quality text (user input), NEVER from
    the model's own output — that creates a degenerate feedback loop."""

    def __init__(self, ngram_model: SmoothedNGramModel):
        self.model = ngram_model

    def learn(self, tokens: List[str]):
        """Add new tokens to the n-gram model incrementally."""
        for token in tokens:
            self.model.vocab.add(token)
            self.model.token_freq[token] += 1
            self.model.unigram_count += 1

        for n in range(1, self.model.max_n + 1):
            if n not in self.model.ngram_contexts:
                self.model.ngram_contexts[n] = {}
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i + n])
                next_tok = tokens[i + n]
                if ngram not in self.model.ngram_contexts[n]:
                    self.model.ngram_contexts[n][ngram] = Counter()
                self.model.ngram_contexts[n][ngram][next_tok] += 1


# ══════════════════════════════════════════════════════════════════════════════
#  MLLM-5-BASE  —  Main Model Class
# ══════════════════════════════════════════════════════════════════════════════

class MLLM5_Atlas:
    """MLLM-5-BASE: Advanced Tiny Language Architecture System.

    Architecture stack:
        1. BPE-lite tokenizer (subword)
        2. Embedding + positional encoding + transformer blocks  (available
           but OFF by default — untrained random weights add noise)
        3. Fast interpolated-backoff n-gram backbone  (direct successor
           lookup — O(k) not O(|V|) per prediction step)
        4. Tool engine (18 tools)
        5. Context memory (user-query-only retrieval — no model output
           contamination)
        6. Online learner (user input only — prevents feedback loop)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_ngram: int = 20,
        beam_width: int = 1,
        context_window: int = 50,
        vocab_size: int = 2000,
        use_transformer: bool = False,   # OFF by default — needs real training
    ):
        # Hyper-parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_ngram = max_ngram
        self.beam_width = beam_width
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.use_transformer = use_transformer

        # Components
        self.tokenizer = BPETokenizer(vocab_size)
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks: List[TransformerBlock] = [
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.ngram_model = SmoothedNGramModel(max_n=max_ngram)
        self.context_memory = ContextMemory(max_turns=context_window)
        self.online_learner = OnlineLearner(self.ngram_model)
        self.tool_engine = ToolEngine

        # State
        self.trained = False
        self.corpus = ""
        self.tokens: List[str] = []
        self.entropy = 0.0
        self.perplexity = 0.0
        self.total_interactions = 0

    # ── TRAINING ─────────────────────────────────────────────────────────

    def train(self, corpus: str, verbose: bool = True):
        """Full training pipeline: tokenize, train n-grams."""
        if verbose:
            print("\n" + "=" * 55)
            print("  MLLM-5-BASE  Training Pipeline")
            print("=" * 55)

        self.corpus = corpus

        # 1. Train BPE tokenizer
        if verbose:
            print("  [1/3] Training BPE tokenizer...")
        self.tokenizer.train(corpus, verbose=verbose)

        # 2. Word-level tokens for n-gram model
        if verbose:
            print("  [2/3] Building word-level token stream...")
        self.tokens = self._word_tokenize(corpus)

        # 3. Train fast n-gram model
        if verbose:
            print("  [3/3] Training n-gram model (interpolated backoff)...")
        self.ngram_model.train(self.tokens)

        self._compute_metrics()
        self.trained = True

        if verbose:
            print(f"\n  Vocabulary size:    {len(self.ngram_model.vocab)}")
            print(f"  Total tokens:       {len(self.tokens)}")
            print(f"  Entropy:            {self.entropy:.4f} bits/token")
            print(f"  Perplexity:         {self.perplexity:.2f}")
            print(f"  Transformer:        {'ON' if self.use_transformer else 'OFF'} (use_transformer={self.use_transformer})")
            print(f"  Max n-gram:         {self.max_ngram}")
            print(f"  Tool count:         18")
            print("=" * 55 + "\n")

    @staticmethod
    def _word_tokenize(text: str) -> List[str]:
        """Simple word + punctuation tokenizer for the n-gram model."""
        return [t for t in re.findall(r'\b\w+\b|[.!?,;:\'"()]', text.lower()) if t.strip()]

    def _compute_metrics(self):
        """Compute entropy and perplexity from token frequencies."""
        freq = self.ngram_model.token_freq
        total = self.ngram_model.unigram_count
        if total == 0:
            return
        entropy_sum = 0.0
        for token, count in freq.items():
            p = count / total
            if p > 0:
                entropy_sum += -p * math.log2(p)
        self.entropy = entropy_sum
        self.perplexity = 2 ** self.entropy

    # ── GENERATION ───────────────────────────────────────────────────────

    def generate_response(self, user_input: str, max_length: int = 25,
                          temperature: float = 0.8) -> str:
        """Generate a response: route to tools first, then generate text.

        Speed notes:
        • Tool routing is fast — regex checks, returns immediately on match.
        • N-gram generation only scores words SEEN after the context, not
          the entire vocabulary.  O(k) per step where k is typically 1-5.
        • Transformer blocks are SKIPPED by default (use_transformer=False)
          because untrained random weights only add noise to temperature.
        """
        # 1. Tool routing (fast path — returns immediately if matched)
        tool_result = self.tool_engine.route(user_input)
        if tool_result is not None:
            return tool_result

        if not self.trained:
            return "Model not trained yet. Please run .train(corpus) first."

        # 2. Build context: user input + relevant past user queries
        #    (NOT model output — prevents degenerate feedback loop)
        extra_context = self.context_memory.retrieve_user_context(user_input, top_k=1)
        context_text = (extra_context + " " + user_input).strip() if extra_context else user_input
        context_tokens = self._word_tokenize(context_text)

        # 3. Optional transformer pass (off by default)
        if self.use_transformer:
            token_ids = self.tokenizer.encode(context_text)
            if token_ids:
                token_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]
                x = self.embedding.forward(token_ids)
                for block in self.transformer_blocks:
                    x = block.forward(x)
                x = self.ln_final.forward(x)
                # Modulate temperature from hidden state
                if x:
                    last_vec = x[-1]
                    mean_val = sum(last_vec) / len(last_vec)
                    temperature = max(0.3, min(1.5, temperature + mean_val * 0.1))

        # 4. Generate via fast n-gram sampling
        #    Don't stop on the FIRST punctuation — generate at least a few
        #    words so the response isn't trivially short.
        response_tokens: List[str] = []
        for step in range(max_length):
            word, _ = self.ngram_model.predict_next(context_tokens, temperature)
            response_tokens.append(word)
            context_tokens.append(word)
            # Allow early stop on sentence-ending punctuation, but only
            # after we've generated at least 4 words
            if word in ['.', '!', '?'] and step >= 3:
                break

        # 5. Format response
        response = " ".join(response_tokens)
        response = response.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        response = response.replace(" ,", ",").replace(" ;", ";").replace(" :", ":")

        # 6. Prepend user input to the response
        #    e.g. input "hello" -> output "hello there, how can i help you."
        #    Strip the echoed input if the n-gram model already repeated it,
        #    then prepend it once cleanly.
        input_lower = user_input.lower()
        input_tokens = self._word_tokenize(user_input)
        input_prefix = " ".join(input_tokens)
        if response.startswith(input_prefix):
            response = response[len(input_prefix):].strip()
        # Prepend the original user input
        response = user_input.strip() + " " + response

        return response if response else "..."

    # ── INTERACTION ──────────────────────────────────────────────────────

    def chat(self, user_input: str, temperature: float = 0.8) -> str:
        """Full chat pipeline: store context, generate, learn from user only."""
        self.total_interactions += 1

        # Generate response
        response = self.generate_response(user_input, temperature=temperature)

        # Store in context memory (both sides for display, but generation
        # only uses the user side — see retrieve_user_context)
        self.context_memory.add(user_input, response)

        # Online learning: ONLY from user input, NOT from model output
        user_tokens = self._word_tokenize(user_input)
        self.online_learner.learn(user_tokens)

        return response

    # ── STATS ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "MLLM-5-BASE",
            "trained": self.trained,
            "vocabulary_size": len(self.ngram_model.vocab),
            "total_tokens": len(self.tokens),
            "entropy_bits": round(self.entropy, 4),
            "perplexity": round(self.perplexity, 2),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_ngram": self.max_ngram,
            "tool_count": 18,
            "use_transformer": self.use_transformer,
            "context_window": self.context_window,
            "total_interactions": self.total_interactions,
            "context_memory_turns": len(self.context_memory.turns),
            "bpe_vocab_size": len(self.tokenizer.vocab),
            "bpe_merges": len(self.tokenizer.merges),
        }

    def get_config(self) -> str:
        """Return a formatted summary of the model configuration."""
        s = self.get_stats()
        return (
            f"Model:              {s['model']}\n"
            f"Trained:            {s['trained']}\n"
            f"Vocabulary:         {s['vocabulary_size']} words\n"
            f"Total tokens:       {s['total_tokens']}\n"
            f"Entropy:            {s['entropy_bits']} bits/token\n"
            f"Perplexity:         {s['perplexity']}\n"
            f"d_model:            {s['d_model']}\n"
            f"Attention heads:    {s['n_heads']}\n"
            f"Transformer layers: {s['n_layers']} (active: {s['use_transformer']})\n"
            f"Feed-forward dim:   {s['d_ff']}\n"
            f"Max n-gram:         {s['max_ngram']}\n"
            f"Tools:              {s['tool_count']}\n"
            f"BPE vocab:          {s['bpe_vocab_size']}\n"
            f"BPE merges:         {s['bpe_merges']}\n"
            f"Interactions:       {s['total_interactions']}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 60)
    print("   MLLM-5-BASE")
    print("   Advanced Tiny Language Architecture System")
    print("   " + "-" * 52)
    print("   Math: 2+2, sqrt(16), sin(pi/4)")
    print("   Tools: base64, hash, units, date, color, roman,")
    print("          stats, combinatorics, geometry, finance,")
    print("          gcd/lcm, equations, text analysis, probability")
    print("   Chat: type naturally for language generation")
    print("   Commands: /stats /config /context /clear /help /quit")
    print("=" * 60 + "\n")


def main():
    print_banner()

    # Initialize model — transformer OFF by default for speed
    # Set use_transformer=True if you have real trained weights
    model = MLLM5_Atlas(
        d_model=64,          # Embedding / hidden dimension
        n_heads=4,           # Number of attention heads
        n_layers=2,          # Transformer blocks (available but off)
        d_ff=256,            # Feed-forward hidden dimension
        max_ngram=20,        # Maximum n-gram order
        beam_width=1,        # 1 = fast sampling (no beam search)
        context_window=50,   # Conversation memory turns
        vocab_size=2000,     # BPE vocabulary size
        use_transformer=False,  # Off — untrained weights add noise
    )

    # Train on default corpus
    model.train(CORPUS, verbose=True)

    print("Ready. Type math expressions, tool commands, or chat.")
    print("Type /help for tools list, /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', '/quit', '/exit', 'bye'):
            print("Goodbye.")
            break
        elif cmd in ('/stats', 'stats'):
            stats = model.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue
        elif cmd in ('/config', 'config'):
            print(model.get_config())
            continue
        elif cmd in ('/context', 'context'):
            recent = model.context_memory.get_recent(5)
            if not recent:
                print("  No conversation history yet.")
            else:
                for i, turn in enumerate(recent, 1):
                    print(f"  [{i}] You: {turn['user']}")
                    print(f"      Atlas: {turn['model']}")
            print()
            continue
        elif cmd in ('/clear', 'clear'):
            model.context_memory.clear()
            print("  Context memory cleared.\n")
            continue
        elif cmd in ('/help', 'help', '!tools', 'tools'):
            print(ToolEngine.route("help"))
            print()
            continue

        response = model.chat(user_input)
        print(f"Atlas: {response}\n")


if __name__ == "__main__":
    main()
