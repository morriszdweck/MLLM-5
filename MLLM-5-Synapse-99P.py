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

2+2 is equal to 4. 

1+1 is equal to 2 

3+3 is equal to 6 

4x3 is equal to 12 

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

Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives. 

Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions. 

Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. 

Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. 

Music theory is the study of theoretical frameworks for understanding the practices and possibilities of music. The Oxford Companion to Music describes three interrelated uses of the term "music theory": The first refers to the "rudiments" needed to understand music notation such as key signatures, time signatures, and rhythmic notation; the second is a study of scholars' views on music from antiquity to the present; the third is a sub-topic of musicology that "seeks to define processes and general principles in music". The musicological approach to theory differs from musical analysis "in that it takes as its starting-point not the individual work or performance but the fundamental materials from which it is built.". 

Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure, machinery, vehicles, electronics, materials, and energy systems. For related terms, see glossary of engineering. 

Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use. 

The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global. 

Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways. 

Computer science is the study of computation, information, and automation. Included broadly in the sciences, computer science spans theoretical disciplines to applied disciplines. An expert in the field is known as a computer scientist. 

Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide. 

Data science is an interdisciplinary academic field that uses statistics, scientific computing, scientific methods, processing, scientific visualization, algorithms, and systems to extract or extrapolate knowledge from potentially noisy, structured, or unstructured data. 

Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. 

A neural network is a group of interconnected units called neurons that send signals to one another. Neurons can be either biological cells or mathematical models. While individual neurons are simple, many of them together in a network can perform complex tasks. There are two main types of neural networks.In neuroscience, a biological neural network is a physical structure found in brains and complex nervous systems – a population of nerve cells connected by synapses. 

In machine learning, an artificial neural network is a mathematical model used to approximate nonlinear functions. Artificial neural networks are used to solve artificial intelligence problems. 

A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems that evolve in ways that may be described as operating on an enormous number of possibilities simultaneously, though still subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks. 

A blockchain is a distributed ledger with growing lists of records (blocks) that are securely linked together via cryptographic hashes. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Since each block contains information about the previous block, they effectively form a chain, with each additional block linking to the ones before it. Consequently, blockchain transactions are resistant to alteration because, once recorded, the data in any given block cannot be changed retroactively without altering all subsequent blocks and obtaining network consensus to accept these changes. 

Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications. 

Network, networking and networked may refer to:. 

An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs. 

Cloud computing is defined by the ISO as "a paradigm for enabling network access to a scalable and elastic pool of shareable physical or virtual resources with self-service provisioning and administration on demand". It is commonly referred to as "the cloud". 

In computing, a database is an organized collection of data or a type of data store based on the use of a database management system (DBMS), the software that interacts with end users, applications, and the database itself to capture and analyze the data. The DBMS additionally encompasses the core facilities provided to administer the database. The sum total of the database, the DBMS and the associated applications can be referred to as a database system. Often the term "database" is also used loosely to refer to any of the DBMS, the database system or an application associated with the database. 

Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene. 

Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences. 

Medicine is the science and practice of caring for patients, managing the diagnosis, prognosis, prevention, treatment and palliation of their injury or disease, while promoting their health. Medicine encompasses a variety of health care practices which evolved to maintain and restore health through the prevention and treatment of illness. Contemporary medicine applies biomedical sciences, biomedical research, genetics, and medical technology to diagnose, treat, and prevent injury and disease, typically through various pharmaceuticals or surgery, but also through therapies such as psychotherapy, external splints and traction, medical devices, biologics, and ionizing radiation, amongst others. 

Public Health may refer to:Public health, promoting health through organized efforts and informed choices of society and individuals 

Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people. 

Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics. 

Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.". 

Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance. 

Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce. 

Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations. 

Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them.  

Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change.  

Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment.  

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

Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community.  

Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping.  

A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged.  

Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan. 

Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment.  

Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well.  

Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events. 

Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies.  

Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks.  

Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission.  

A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage. 

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

Quinoa is a flowering plant in the amaranth family. It is a herbaceous annual plant grown as a crop primarily for its edible seeds. 

Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally.  

The lentil is an annual legume grown for its lens-shaped edible seeds or pulses, also called lentils.  

The chickpea or chick pea is an annual legume of the family Fabaceae, subfamily Faboideae, cultivated for its edible seeds.  

The kidney bean is a variety of the common bean ; it has such a common name owing to its resemblance to a human kidney. 

Tofu  or bean curd is a food prepared by pressing the curds of coagulated soy milk into solid white blocks of varying softness: silken, soft, firm, and extra firm. 

Tempeh or tempe is a traditional Indonesian food made from fermented soybeans.  

The chicken is a domesticated form of the red junglefowl, originally native to Southeast Asia.  

Beef is the culinary name for meat from cattle.  

Pork is the culinary name for the meat of the pig.  

Turkey, officially the Republic of Türkiye, is a country mainly located in Anatolia in West Asia, with a smaller part called East Thrace in Southeast Europe.  

Salmon are any of several commercially important species of euryhaline ray-finned fish from the genera Salmo and Oncorhynchus of the family Salmonidae. 

A tuna is a saltwater fish that belongs to the tribe Thunnini, a subgrouping of the Scombridae (mackerel) family.  

A shrimp is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion. 

Crabs are decapod crustaceans, either the Brachyura or various groups within the closely related Anomura. 

Lobsters are malacostracan decapod crustaceans of the family Nephropidae or its synonym Homaridae.  

An egg is an organic vessel in which an embryo begins to develop. 

Milk is a usually white liquid food produced by the mammary glands of lactating mammals.  

Cheddar cheese is a natural cheese that is relatively hard, off-white, and sometimes sharp-tasting.  

Mozzarella is a semi-soft non-aged cheese prepared using the pasta filata ('stretched-curd') method.  

Yogurt is a food produced by bacterial fermentation of milk.  

Butter is a dairy product made from the fat and protein components of churned cream.  

The almond is a species of tree from the genus Prunus.  

A walnut is the edible seed of any tree of the genus Juglans, particularly the Persian or English walnut, Juglans regia.  

Cashew is the common name of a tropical evergreen tree Anacardium occidentale, in the family Anacardiaceae.  

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

Unification or unification theory may refer to:. 

A civilization is any complex society characterized by the development of the state, social stratification, urbanization, and symbolic systems of communication. 

Prehistory, sometimes referred to as pre-literary history, is the period of human history between the first known use of stone tools by hominins c. 3.3 million years ago and the beginning of recorded history. 

Arithmetic is an elementary branch of mathematics that deals with numerical operations like addition, subtraction, multiplication, and division.  

Algebra is a branch of mathematics that deals with abstract systems, known as algebraic structures, and the manipulation of expressions within those systems.  

Geometry is a branch of mathematics concerned with properties of space such as the distance, shape, size, and relative position of figures.  

Trigonometry is a branch of mathematics concerned with relationships between angles and side lengths of triangles.  

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

Computer numerical control (CNC) or CNC machining is the automated control of machine tools by a computer.  

Dynamics or dynamic may refer to:. 

Chaos or CHAOS may refer to:. 

In mathematics, a manifold is a topological space that locally resembles Euclidean space near each point.  

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

Anatomy is the branch of morphology concerned with the study of the internal and external structure of organisms and their parts.  

In meteorology, an anemometer is a device that measures wind speed and direction.  

Angiogenesis is the physiological process through which new blood vessels form from pre-existing vessels. 

Anomaly, The Anomaly or Anomalies may refer to:. 

An anode usually is an electrode of a polarized electrical device through which conventional current enters the device.  

An antibody (Ab), or immunoglobulin (Ig), is a large protein belonging to the immunoglobulin superfamily which is used by the immune system to identify and neutralize antigens. 

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

Atomism is a natural philosophy proposing that the physical universe is composed of fundamental indivisible components known as atoms. 

Atrophy is the partial or complete wasting away of a part of the body.  

An aurora is a natural light display in Earth's sky, predominantly observed in high-latitude regions around the Arctic and Antarctic.  

Autocracy is a form of government in which absolute power is held by one person, known as an autocrat.  

In developmental psychology and moral, political, bioethical philosophy, autonomy is the capacity to make an informed, uncoerced decision.  

Aviation includes the activities surrounding mechanical flight and the aircraft industry.  

An axon is a long slender projection of a nerve cell or neuron found in most animals that typically conducts electrical impulses known as action potentials away from the nerve cell body.  

Bacteria are ubiquitous, mostly free-living organisms often consisting of one biological cell.  

Bandwidth commonly refers to:Bandwidth or analog bandwidth, frequency bandwidth, or radio bandwidth, a measure of the width of a frequency range 

A barometer is a scientific instrument that is used to measure air pressure.  

Basalt is an aphanitic (fine-grained) extrusive igneous rock formed from the rapid cooling of low-viscosity lava rich in magnesium and iron. 

Biodiversity is the variability of life on Earth.  

A biome is a distinct geographical region with specific climate, vegetation, animal life, and an ecosystem.  

Biometrics are body measurements and calculations related to human characteristics and features.  

A biomolecule or biological molecule is loosely defined as a molecule produced by a living organism and essential to one or more typically biological processes.  

The biosphere, also called the ecosphere, is the worldwide sum of all ecosystems.  

In telecommunications and computing, bit rate is the number of bits that are conveyed or processed per unit of time.  

Botany, also called phytology or plant science, is the branch of natural science and biology that studies plants. 

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

Continuum may refer to:Continuum (measurement), theories or models that explain gradual transitions from one condition to another without abrupt changes. 

Convection is single-phase or multiphase fluid flow that occurs spontaneously through the combined effects of material property heterogeneity and body forces on a fluid.  

Corrosion is a natural process that converts a refined metal into a more chemically stable oxide.  

where ρ is the density, m is the mass, and V is the volume.  

Dentition pertains to the development of teeth and their arrangement in the mouth.  

Deposition may refer to:Deposition (law), taking testimony outside of court 

In mathematics, the derivative is a fundamental tool that quantifies the sensitivity to change of a function's output with respect to its input.  

Desalination is the artificial process by which salt water is converted to fresh water. 

Desertification is a type of gradual land degradation of fertile land into arid desert. 

Determinism is the metaphysical view that all events within the universe can occur only in one possible way.  

Deviation may refer to:. 

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

Fusion, or synthesis, is the process of combining two or more distinct entities into a new whole. 

A galaxy is a system of stars, stellar remnants, interstellar gas, dust, and dark matter bound together by gravity.  

Gallium is a chemical element; it has symbol Ga and atomic number 31.  

Gametogenesis is a biological process by which diploid or haploid precursor cells undergo cell division and differentiation to form mature haploid gametes.  

A gantry is an overhead bridge-like structure supporting equipment such as a crane, signals, or cameras. 

Gasification is a process that converts biomass- or fossil fuel-based carbonaceous materials into gases. 

Gastronomy is the study of the relationship between food and culture. 

Genealogy is the study of families, family history, and the tracing of their lineages.  

Geodesy or geodetics is the science of measuring and representing the geometry, gravity, and spatial orientation of the Earth. 

A geoglyph is a large design or motif – generally longer than 4 metres (13 ft) – produced on the ground by durable elements of the landscape. 

Geophysics is a physical science concerned with the processes and properties of Earth and its surrounding space environment. 

Germination is the process by which an organism grows from a seed or spore.  

A geyser is a spring with an intermittent water discharge ejected turbulently and accompanied by steam.  

A glacier is a persistent body of dense ice, a form of rock, that is constantly moving downhill under its own weight.  

Globalism has multiple meanings.  

A glyph is any kind of purposeful mark.  

Granite is a coarse-grained (phaneritic) intrusive igneous rock composed mostly of quartz, alkali feldspar, mica and plagioclase.  

In physics, gravity, also known as gravitation or a gravitational interaction, is a fundamental interaction. 

A greenhouse is a structure that is designed to regulate the temperature and humidity of the environment inside.  

In ecology, habitat refers to the array of resources, biotic factors that are present in an area. 

The halogens are a group in the periodic table consisting of six chemically related elements. 

In music, harmony is the concept of combining different sounds in order to create new, distinct musical ideas.  

Harvesting is the process of collecting plants, animals, or fish as food. 

Hematology is the branch of medicine concerned with the study of the cause, prognosis, treatment, and prevention of diseases related to blood.  

A herbivore is an animal anatomically and physiologically evolved to feed on plants. 

A hierarchy is an arrangement of items that are represented as being "above", "below", or "at the same level as" one another.  

Holography is a technique that allows a wavefront to be recorded and later reconstructed.  

The Hominidae, whose members are known as the great apes, are a taxonomic family of primates. 

Humidity is the concentration of water vapor present in the air.  

Hybridization may refer to:Hybridization (biology), the process of combining different varieties of organisms to create a hybrid 

Hydration may refer to:Hydrate, a substance that contains water 

Hydraulics is a technology and applied science using engineering, chemistry, and other sciences involving the mechanical properties and use of liquids. 

In organic chemistry, a hydrocarbon is an organic compound consisting entirely of hydrogen and carbon.  

Hydrology is the scientific study of the movement, distribution, and management of water on Earth. 

A hypothesis is a proposed explanation for a phenomenon.  

An iceberg is a piece of fresh water ice more than 15 meters long that has broken off a glacier or an ice shelf. 

Iconography, as a branch of art history, studies the identification, description and interpretation of the content of images. 

An ideology is a set of beliefs or values attributed to a person or group of persons. 

Ignition may refer to:. 

Illumination may refer to:. 

Imaging is the process of creating visual representations of objects, scenes, or phenomena.  

Immigration is the international movement of people to a destination country of which they are not usual residents. 

Immunity may refer to:. 

Implosion can refer to:. 

The word incubation may refer to:. 

Index may refer to:. 

Induction or inductive may refer to:. 

Inertia is the natural tendency of objects in motion to stay in motion and objects at rest to stay at rest. 

Inferences are steps in logical reasoning, moving from premises to logical consequences. 

In economics, inflation is an increase in the average price of goods and services in terms of money.  

Infrared is electromagnetic radiation (EMR) with wavelengths longer than that of visible light but shorter than microwaves.  

In law and conflict of laws, domicile is relevant to an individual's "personal law". 

Injection or injected may refer to:. 

Innovation is the practical implementation of ideas that result in the introduction of new goods or services. 

Insects are hexapod invertebrates of the class Insecta.  

Insertion may refer to:Insertion (anatomy), the point of a tendon or ligament onto the skeleton. 

Insomnia, also known as sleeplessness, is a sleep disorder causing difficulty falling asleep or staying asleep. 

An inspection is, most generally, an organized examination or formal evaluation exercise.  

In dynamical systems instability means that some of the outputs or internal states increase with time, without bounds.  

Instrumentation is a collective term for measuring instruments, used for indicating, measuring, and recording physical quantities.  

Insulation may refer to:. 

Intensity may refer to:. 

Interaction is action that occurs between two or more entities. 

Interface or interfacing may refer to:. 

Interference is the act of interfering, invading, or poaching.  

The intertidal zone or foreshore is the area above water level at low tide and underwater at high tide. 

Inversion or inversions may refer to:. 

Ionization or ionisation is the process by which an atom or a molecule acquires a negative or positive charge by gaining or losing electrons. 

Irradiation is the process by which an object is exposed to radiation.  

Irrigation is the practice of applying controlled amounts of water to land to help grow crops. 

Isobar may refer to:Isobar (meteorology), a line on a map or chart connecting points of equal atmospheric pressure reduced to sea level. 

Isotopes are distinct nuclear species of the same chemical element.  

Iteration means repeating a process to generate a sequence of outcomes.  

Jet streams are fast flowing, narrow air currents in the atmosphere.  

Jurisdiction is the legal term for the legal authority held by a legal entity to enact justice.  

A karyotype is the general appearance of the complete set of chromosomes in the cells of a species. 

Kinetics may refer to:. 

In Greek mythology, the Labyrinth is an elaborate, confusing structure designed and built by the mythological artificer Daedalus. 

Lactation describes the secretion of milk from the mammary glands. 

Lamination is the technique/process of manufacturing a material in multiple layers. 

A landscape is the visible features of an area of land, its landforms, and how they integrate with natural or human-made features. 

Landslides, also known as landslips, rockslips or rockslides, are several forms of mass wasting. 

Language is a structured system of communication that consists of grammar and vocabulary.  

Lattice may refer to:. 

In geography, latitude is a geographic coordinate that specifies the north-south position of a point on the surface of the Earth. 

Lava is molten or partially molten rock (magma) that has been expelled from the interior of a terrestrial planet. 

Layering can refer to:Layering (horticulture), a means of vegetative propagation 

Leadership, is defined as the ability of an individual, group, or organization to influence, or guide other individuals, teams, or organizations. 

Legislation is the process or result of enrolling, enacting, or promulgating laws by a legislature, parliament, or analogous governing body.  

Lenticular is an adjective often relating to lenses.  

Lepidoptera or lepidopterans are an order of winged insects which include butterflies and moths.  

A lexicon is the vocabulary of a language or branch of knowledge.  

In lunar astronomy, libration is the cyclic variation in the apparent position of the Moon. 

A lichen is a hybrid colony of algae or cyanobacteria living symbiotically among filaments of multiple fungus species. 

Lifespan or life span may refer to:Lifespan (film), 1976 film starring Klaus Kinski 

A ligament is a type of fibrous connective tissue in the body that connects bones to other bones.  

Lignite, often called brown coal, is a soft, brown, combustible sedimentary rock formed from naturally compressed peat.  

Limnology is the study of inland aquatic ecosystems.  

A lithosphere is the rigid outermost rocky shell of a terrestrial planet or natural satellite.  

Longitude is a geographic coordinate that specifies the east-west position of a point on the surface of the Earth. 

Luminosity is an absolute measure of radiated electromagnetic energy per unit time. 

A lymphocyte is a type of white blood cell (leukocyte) in the immune system of most vertebrates.  

The Fauna is the whole of animal life present in a particular region or time.  

Magnetism is the class of physical attributes that occur through a magnetic field. 

Magnitude may refer to:. 

A mammal is a vertebrate animal of the class Mammalia.  

A manuscript was, traditionally, any document written by hand or typewritten. 

Maritime may refer to:. 

A massacre is an event of killing defenseless human beings or other animals.  

In philosophy and metaphysics, materialism is a form of monism holding that matter is the fundamental substance of nature. 

Mechanism may refer to:Mechanism (economics), a set of rules for a game designed to achieve a certain outcome 

A megalith is a large stone that has been used to construct a prehistoric structure or monument. 

Melanin is a family of biomolecules organized as oligomers or polymers, which among other functions provide the pigments of many organisms.  

A membrane is a selective barrier; it allows some things to pass through but stops others.  

Metabolism refers to the set of life-sustaining chemical reactions that occur within living organisms.  

Metallurgy is a domain of materials science and engineering that studies the physical and chemical behavior of metallic elements. 

A meteorite is a rock that originated in outer space and has fallen to the surface of a planet or moon.  

Meteorology is the scientific study of the Earth's atmosphere and short-term atmospheric phenomena. 

A microorganism, or microbe, is an organism of microscopic size. 

A microclimate refers to localized atmospheric conditions in the near-surface layer. 

Microfauna are microscopic animals and organisms that exhibit animal-like qualities. 

Weightlessness is the complete or near-complete absence of the sensation of weight. 

Microscopy is the technical field of using microscopes to view subjects too small to be seen with the naked eye.  

A militia is a military or paramilitary force that comprises civilian members. 

Mineralogy is a subject of geology specializing in the scientific study of the chemistry, crystal structure, and physical properties of minerals. 

In visual arts, music, and other media, minimalism is an art movement that emerged in the post-World War II era in Western art.  

A mirage is a naturally occurring optical phenomenon in which light rays bend via refraction to produce a displaced image of distant objects or the sky.  

Mitosis is a part of the cell cycle in eukaryotic cells in which replicated chromosomes are separated into two new nuclei.  

In chemistry, a mixture is a material made up of two or more different chemical substances which can be separated by physical method.  

Signal modulation is the process of varying one or more properties of a periodic waveform in electronics and telecommunication. 

In Newtonian mechanics, momentum is the product of the mass and velocity of an object.  

A monolith is a geological feature consisting of a single massive stone or rock. 

A monsoon is traditionally a seasonal reversing wind accompanied by corresponding changes in precipitation. 

Morphology, from the Greek and meaning "study of shape", may refer to:. 

Mortality may refer to:Fish mortality, a parameter used in fisheries population dynamics. 

Generally, a motif is a recurring element or theme in a work of art or media. 

In biology, a mutation is an alteration in the nucleic acid sequence of the genome of an organism, virus, or extrachromosomal DNA.  

Mycology is the branch of biology concerned with the study of fungi. 

Nanofibers are fibers with diameters in the nanometer range.  

A nanoparticle or ultrafine particle is a particle of matter 1 to 100 nanometres (nm) in diameter.  

A narrative, story, or tale is any account of a series of related events or experiences. 

Not to be confused with Naturism. 

A nebula is a distinct luminescent part of interstellar medium. 

Necrosis is a form of cell injury which results in the premature death of cells in living tissue by autolysis.  

The Neolithic or New Stone Age is an archaeological period, the final division of the Stone Age. 

In common terminology, a baby is the very young offspring of adult human beings. 

Neoteny, also called juvenilization, is the delaying or slowing of the physiological, or somatic, development of an organism. 

Nephrology is a specialty for both adult internal medicine and pediatric medicine that concerns the study of the kidneys. 

Neuralgia is pain in the distribution of a nerve or nerves. 

A neutrino is an elementary particle that interacts via the weak interaction and gravity.  

Niche may refer to:. 

Nitrate is a polyatomic ion with the chemical formula NO−3.  

Nitrogen is a chemical element; it has symbol N and atomic number 7.  

Nomads are communities without fixed habitation who regularly move to and from areas.  

In linguistics and semiotics, a notation system is a system of graphics or symbols. 

Nucleus is a Latin word for the seed inside a fruit.  

A nutrient is a substance used by an organism to survive, grow and reproduce.  

Obsidian is a naturally occurring volcanic glass formed when lava extruded from a volcano cools rapidly with minimal crystal growth.  

The sense of smell, or olfaction, is the special sense through which smells are perceived.  

Oligarchy is a form of government in which power rests with a small number of people.  

An oligopoly is a market in which pricing control lies in the hands of a few sellers. 

An ombudsman is a government official who investigates and tries to resolve complaints. 

Ontology is the philosophical study of being.  

Opacity is the measure of impenetrability to electromagnetic or other kinds of radiation. 

In genetics, an operon is a functioning unit of DNA containing a cluster of genes under the control of a single promoter.  

In celestial mechanics, an orbit is the curved trajectory of an object under the influence of an attracting force.  

An organism is any living thing that functions as an individual.  

Osmosis is the spontaneous net movement of solvent molecules through a selectively permeable membrane. 

In epidemiology, an outbreak is a sudden increase in occurrences of a disease. 

Redox is a type of chemical reaction in which the oxidation states of the reactants change.  

Paleoclimatology is the scientific study of climates predating the invention of meteorological instruments. 

Palaeography (UK) or paleography (US) is the study and academic discipline of historical writing systems.  

A pandemic is an epidemic of an infectious disease that has a sudden increase in cases and spreads across a large region. 

A panorama is any wide-angle view or representation of a physical space. 

A paradox is a logically self-contradictory statement or a statement that runs contrary to one's expectation.  

Parallax is a displacement or difference in the apparent position of an object viewed along two different lines of sight. 

Parasitism is a close relationship between species, where one organism, the parasite, lives on or inside another organism, the host. 

In the physical sciences, a particle is a small localized object which can be described by several physical or chemical properties. 

In biology, a pathogen, in the oldest and broadest sense, is any organism, agent or micro-organism that can produce disease.  

Pathology is the study of disease.  

Pedagogy, most commonly understood as the approach to teaching, is the theory and practice of learning. 

A peninsula is a landform that extends from a mainland, is connected to the mainland on only one side, and is mostly surrounded by water.  

In geometry, a pentagon is any five-sided polygon or 5-gon.  

Peptides are short chains of amino acids linked by peptide bonds.  

Perception is the organization, identification, and interpretation of sensory information. 

In physics, chemistry, and materials science, percolation refers to the movement and filtering of fluids through porous materials.  

Permafrost is soil or underwater sediment which continuously remains below 0 °C (32 °F) for two years or more. 

Permeability, permeable, and semipermeable may refer to:. 

In mathematics, a permutation of a set can mean one of two different things:an arrangement of its members in a sequence or linear order. 

Pesticides are substances that are used to control pests.  

Petrology is the branch of geology that studies rocks, their mineralogy, composition, texture, structure and the conditions under which they form.  

Phagocytes are cells that protect the body by ingesting harmful foreign particles, bacteria, and dead or dying cells.  

In genetics, the phenotype is the set of observable characteristics or traits of an organism.  

A phoneme is a set of similar speech sounds that are perceptually regarded by the speakers of a language as a single basic sound. 

A photon is an elementary particle that is a quantum of the electromagnetic field. 

The photosphere is a star's outer shell from which light is radiated.  

In biology, a phylum is a level of classification, or taxonomic rank, that is below kingdom and above class.  

A pinnacle is an architectural element originally forming the cap or crown of a buttress or small turret. 

A pipeline is a system of pipes for long-distance transportation of a liquid or gas. 

A piston is a component of reciprocating engines, reciprocating pumps, gas compressors, hydraulic cylinders and pneumatic cylinders. 

Plankton are organisms that drift in water but are unable to actively propel themselves against currents.  

In geology and physical geography, a plateau, also called a high plain or a tableland, is an area of highland. 

The Pleistocene is the geological epoch that lasted from c. 2.58 million to 11,700 years ago. 

The Pliocene is the epoch in the geologic time scale that extends from 5.33 to 2.58 million years ago (Ma).  

Plutonium is a chemical element; it has symbol Pu and atomic number 94.  

Pneumatics is the use of gas or pressurized air to create mechanical motion in mechanical systems. 

Polarization or polarisation may refer to:. 

A polymer is a substance or material that consists of very large molecules, or macromolecules. 

In biology, a population of organisms is a group of individuals of the same species. 

Porosity or void fraction is a measure of the void spaces in a material. 

Portfolio may refer to:. 

Positioning may refer to:Positioning (marketing), creating an identity in the minds of a target market 

Pottery is the process and the products of forming vessels and other objects with clay and other raw materials. 

In an aqueous solution, precipitation is the "sedimentation of a solid material from a liquid solution". 

Pressure is the force applied perpendicular to the surface of an object per unit area over which that force is distributed.  

Primatology is the scientific study of primates.  

Procedure may refer to:Medical procedure 

Propagation can refer to:Chain propagation in a chemical reaction mechanism 

Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues.  

Protocol may refer to:. 

A proton is a stable subatomic particle, symbol p, H+, or 1H+ with a positive electric charge of +1 e (elementary charge).  

Provenance is the chronology of the ownership, custody or location of a historical object.  

In computer science, pseudocode is a description of the steps in an algorithm using a mix of conventions of programming languages. 

Pseudoscience consists of statements, beliefs, or practices that claim to be scientific or factual but are inherently incompatible with the scientific method.  

Psychometrics is a field of study within psychology concerned with the theory and technique of measurement.  

Pterosaurs are an extinct clade of flying reptiles in the order Pterosauria.  

A pulley is a wheel on an axle or shaft enabling a taut cable or belt passing over the wheel to move and change direction. 

In physics, a quantum is the minimum amount of any physical entity involved in an interaction.  

A quarantine is a restriction on the movement of people, animals, and goods which is intended to prevent the spread of disease or pests.  

A quarry is a type of rock and earth materials. 

A quasar is an extremely luminous active galactic nucleus (AGN).  

A quorum is the minimum number of members of a group necessary to constitute the group at a meeting.  

In physics, radiation is the emission or transmission of energy in the form of waves or particles through space or a material medium.  

Radiometry is a set of techniques for measuring electromagnetic radiation, including visible light.  

Rainforests are forests characterized by a closed and continuous tree canopy, moisture-dependent vegetation, the presence of epiphytes and lianas and the absence of wildfire.  

Reaction may refer to a process or to a response to an action, event, or exposure. 

Reactor may refer to:. 

In chemistry, a reagent or analytical reagent is a substance or compound added to a system to cause a chemical reaction. 

In economics, a recession is a business cycle contraction that occurs when there is a period of broad decline in economic activity.  

Reconstruction may refer to:. 

In physics, refraction is the redirection of a wave as it passes from one medium to another.  

Regolith is a blanket of unconsolidated, loose, heterogeneous superficial deposits covering solid rock.  

In behavioral psychology, reinforcement refers to consequences that increase the likelihood of an organism's future behavior. 

Relativity may refer to:. 

Remedy, Remedies, The Remedy or Remediation may refer to:. 

Reproduction is the biological process by which new individual organisms – offspring – are produced from their parent or parents.  

Resonance is a phenomenon that occurs when an object or system is subjected to an external force or vibration whose frequency matches a resonant frequency of the system.  

Respiration may refer to:. 

The retina is the innermost, light-sensitive layer of tissue of the eye of most vertebrates and some molluscs.  

Rheology is the study of the flow of matter, primarily in a fluid state. 

Otorhinolaryngology is a surgical subspecialty within medicine that deals with the surgical and medical management of conditions of the head and neck.  

Rigid or rigidity may refer to:. 

A streambed or stream bed is the bottom of a stream or river and is confined within a channel or the banks of the waterway.  

A robot is a machine, especially one programmable via a computer, capable of automatically carrying out a complex series of actions.  

Rotation, rotational or rotary motion is the movement of an object that leaves at least one point unchanged.  

Runoff, run-off or RUNOFF may refer to:Runoff (hydrology), the flow of water over land 

Sacrifice is an act or offering made to a deity.  

Salinity is the saltiness or amount of salt dissolved in a body of water.  

Sampling may refer to:Sampling, converting a continuous signal into a discrete signal 

Sandstone is a clastic sedimentary rock composed mainly of sand-sized silicate grains. 

Sanitation refers to public health conditions related to clean drinking water and treatment and disposal of human excreta and sewage.  

Saturation, saturated, unsaturation or unsaturated may refer to:. 

Scaffolding, also called scaffold or staging, is a temporary structure used to support a work crew and materials. 

In physics, scattering is a wide range of physical processes where moving particles or radiation of some form are forced to deviate from a straight trajectory. 

A schism is a division between people, usually belonging to an organization, movement, or religious denomination.  

Scholasticism was a medieval European philosophical movement or methodology. 

The seabed is the bottom of the ocean.  

Seagrasses are the only flowering plants which grow in marine environments.  

A seamount is a large submarine landform that rises from the ocean floor without reaching the water surface. 

Secretion is the movement of material from one point to another. 

Sediment is a solid material made of loose particles that is transported to a new location where it is deposited.  

Seismicity is a measure encompassing earthquake occurrences, mechanisms, and magnitude at a given geographical location.  

Semantics is the study of linguistic meaning.  

Senescence or biological aging is the gradual deterioration of functional characteristics in living organisms.  

Sensory may refer to:. 

Separation may refer to:. 

Serology is the scientific study of serum and other body fluids.  

Settlement may refer to:Human settlement, a community where people live 

In computer graphics, a shader is a programmable operation which is applied to data as it moves through the rendering pipeline.  

Shear may refer to:. 

Shellfish, in colloquial and fisheries usage, are exoskeleton-bearing aquatic invertebrates used as food. 

A signal is both the process and the result of transmission of data over some media accomplished by embedding some variation.  

A silicate is any member of a family of polyatomic anions consisting of silicon and oxygen. 

A simulation is an imitative representation of a process or system that could exist in the real world.  

Sintering or frittage is the process of compacting and forming a solid mass of material by pressure or heat without melting it to the point of liquefaction.  

Skepticism (US) or scepticism (UK) is a questioning attitude or doubt toward knowledge claims that are seen as mere belief or dogma.  

A skylight is a light-permitting structure or window, usually made of transparent or translucent glass. 

Slippage may refer to:Degree of slipping or loosening as result of slipperiness 

Snowpack is an accumulation of snow that compresses with time and melts seasonally. 

A solstice is the time when the Sun reaches its most northerly or southerly excursion relative to the celestial equator on the celestial sphere.  

A solvent is a substance that dissolves a solute, resulting in a solution.  

Sonar is a technique that uses sound propagation to navigate, measure distances (ranging), communicate with or detect objects on or under the surface of the water. 

Sorption is a physical and chemical process by which one substance becomes attached to another.  

A soundscape is the acoustic environment as perceived by humans, in context.  

Sovereignty is generally defined as supreme, independent control and lawmaking authority over a territory.  

Spectroscopy is the field of study that measures and interprets electromagnetic spectra as it interacts with matter.  

A spectrum is a set of related ideas, objects, or properties whose features overlap such that they blend to form a continuum.  

In finance, speculation is the purchase of an asset with the hope that that asset will become more valuable in a brief amount of time. 

In mathematics, a spiral is a curve which emanates from a point, moving further away as it revolves around the point.  

In biology, a spore is a unit of sexual or asexual reproduction that may be adapted for dispersal and for survival. 

Stability may refer to:. 

A stalagmite is a type of rock formation that rises from the floor of a cave due to the accumulation of material deposited on the floor from ceiling drippings.  

A stalactite is a mineral formation that hangs from the ceiling of caves, hot springs, or man-made structures such as bridges and mines.  

Standardization or standardisation is the process of implementing and developing technical standards based on the consensus of different parties. 

Starburst most often refers to:Starburst region, a generic term to describe a region of space with a much higher than normal star formation 

A state is a political entity that regulates society and the population within a definite territory.  

In mathematics and statistics, a stationary process is a stochastic process whose statistical properties, such as mean and variance, do not change over time.  

In physical geography, a steppe is an ecoregion characterized by grassland plains without closed forests except near rivers and lakes. 

Sterile or sterility may refer to:Asepsis, a state of being free from biological contaminants 

Stimulants are a class of psychoactive drugs that increase alertness.  

Stoichiometry is the relationships between the quantities of reactants and products before, during and after chemical reactions. 

Stratification may refer to:. 

The stratosphere is the second-lowest layer of the atmosphere of Earth, located above the troposphere and below the mesosphere.  

Stress may refer to:. 

Subduction is a geological process in which the oceanic lithosphere and some continental lithosphere is recycled into the Earth's mantle at the convergent boundaries between tectonic plates.  

Substrate may refer to:. 

Succession is the act or process of following in order or sequence. 

A supernova is a powerful and luminous explosion of a star.  

Supremacy may refer to:. 

A surface, as the term is most generally used, is the outermost or uppermost layer of a physical object.  

Surge means a sudden transient rush or flood, and may refer to:. 

Survival or survivorship, the act of surviving, is the propensity of something to continue existing despite conditions that might kill or destroy it.  

Symbiosis is any close and long-term biological interaction between two organisms of different species.  

Synthesis or synthesize may refer to:. 

In classical Greek mythology, Syrinx was an Arcadian nymph and a follower of Artemis, known for her chastity.  

Taxonomy is a practice and science concerned with classification or categorization.  

Technocracy is an expert-based type of governance.  

Telemetry is the in situ collection of measurements or other data at remote points and their automatic transmission to receiving equipment (telecommunication) for monitoring.  

In psychology, temperament broadly refers to consistent individual differences in behavior that are biologically based. 

Temperature is a numerical expression of hotness or coldness.  

Tension may refer to:. 

Ternary or trinary is an adjective meaning "composed of three items".  

A territory is an area of land, sea, or space, belonging or connected to a particular country, person, or animal.  

A tessellation or tiling is the covering of a surface, often a plane, using one or more geometric shapes, called tiles, with no overlaps and no gaps.  

A thermocline is a distinct layer based on temperature within a large body of fluid with a high gradient of distinct temperature differences associated with depth.  

A thermometer is a device that measures temperature or temperature gradient.  

Threshold may refer to:. 

Tidal is the adjectival form of tide. 

Titration is a common laboratory method of quantitative chemical analysis to determine the concentration of an identified analyte.  

A tornado, also known as a twister, is a rapidly rotating column of air that extends vertically from the surface of the Earth to the base of a cumulonimbus or cumulus cloud.  

Toxicity is the degree to which a chemical substance or a particular mixture of substances can damage an organism.  

A trajectory is the path an object takes through its motion over time.  

Transcription refers to the process of converting sounds into letters or musical notes. 

A transducer is a device that usefully converts energy from one form to another.  

Transfer may refer to:. 

A transistor is a semiconductor device used to amplify or switch electrical signals and power.  

Transmutation may refer to:. 

Transport or transportation is the intentional movement of humans, animals, and goods from one location to another.  

A tremor is an involuntary, somewhat rhythmic muscle contraction and relaxation involving oscillations or twitching movements of one or more body parts.  

A tributary, or an affluent, is a stream or river that flows into a larger stream, river, or a lake.  

The tropics are the region of Earth surrounding the equator, where the Sun may shine directly overhead.  

A tsunami is a series of waves in a water body caused by the displacement of a large volume of water. 

In fluid dynamics, turbulence or turbulent flow is fluid motion exhibiting chaotic changes in pressure and flow velocity.  

Ultrasound is sound with frequencies greater than 20 kilohertz.  

Ultraviolet radiation (UV) is electromagnetic radiation of wavelengths of 100–400 nanometers, shorter than that of visible light, but longer than X-rays.  

Uncertainty or incertitude refers to situations involving imperfect or unknown information.  

Undercurrent is a flow of water below the surface:In an ocean, a subsurface current. 

Underground most commonly refers to:Subterranea (geography), the regions beneath the surface of the Earth. 

Uplift may refer to:. 

Uranium is a chemical element; it has symbol U and atomic number 92.  

Urbanism is the scientific study of how inhabitants of urban areas, such as towns and cities, interact with the built environment.  

Vacancy or No Vacancy may refer to:. 

A vaccine is a biological preparation that provides active acquired immunity to a particular infectious or malignant disease.  

A vacuole is a membrane-bound organelle which is present in plant and fungal cells and some protist, animal, and bacterial cells.  

Valence or valency may refer to:. 

Vaporization of an element or compound is a phase transition from the liquid phase to vapor.  

Variable may refer to:. 

In probability theory and statistics, variance is the expected value of the squared deviation from the mean of a random variable.  

Velocity is a measurement of speed in a certain direction of motion.  

Veneer may refer to:. 

Ventilation may refer to:Ventilation (physiology), the movement of air between the environment and the lungs via inhalation and exhalation 

Vertebrates, also called craniates, are animals with a vertebral column and a cranium.  

In mechanics, vibration is oscillatory motion about an equilibrium point.  

The word Viral means "relating to viruses". 

Virulence is a pathogen's or microorganism's ability to cause damage to a host. 

When two fluid layers move relative to each other, a friction force develops between them and the slower layer acts to slow down the faster layer.  

Volatility or volatile may refer to:. 

A volcano is a vent or fissure in the crust of a planetary-mass object, such as Earth, that allows hot lava, volcanic ash, and gases to escape from a magma chamber below the surface. 

Voltage, also known as (electrical) potential difference, electric pressure, or electric tension, is the difference in electric potential between two points.  

In fluid dynamics, a vortex is a region in a fluid in which the flow revolves around an axis line. 

In physics and mathematics, wavelength or spatial period of a wave or periodic function is the distance over which the wave's shape repeats.  

Weathering is the deterioration of rocks, soils and minerals through contact with water, atmospheric gases, sunlight, and biological organisms.  

A wetland is a distinct semi-aquatic ecosystem whose groundcovers are flooded or saturated in water. 

A whirlpool is a body of rotating water produced by opposing currents or a current running into an obstacle.  

Windfall or Windfalls may refer to:. 

A windmill is a machine operated by the force of wind acting on vanes or sails to mill grain (gristmills), pump water, generate electricity, or drive other machinery. 

A woodland is, in the broad sense, land covered with woody plants. 

In macroeconomics, the workforce or labour force is the sum of people either working or looking for work :. 

A xenolith is a rock fragment that becomes enveloped in a larger rock during the latter's development and solidification.  

Xenon is a chemical element; it has symbol Xe and atomic number 54.  

Yield may refer to:. 

Zeolites are a group of several microporous, crystalline aluminosilicate minerals commonly used as commercial adsorbents and catalysts.  

In European tradition, a zephyr is a light wind or a west wind, named after Zephyrus, the Greek god or personification of the west wind. 

Zoology is the scientific study of animals.  

Abbot is an ecclesiastical title given to the head of an independent monastery for men in various Western Christian traditions.  

An abdomen is the front part of the torso between the thorax (chest) and pelvis in humans and in other vertebrates.  

Abjuration is the solemn repudiation, abandonment, or renunciation by or upon oath. 

Abolitionism, or the abolitionist movement, is the political movement to end slavery and liberate enslaved individuals around the world.  

An abscess is a collection of pus that has built up within the tissue of the body, usually caused by bacterial infection.  

Absorption may refer to:. 

Abundance may refer to:. 

Abyss may refer to:. 

An acetate is a salt formed by the combination of acetic acid with a base.  

Acidification may refer to:Ocean acidification, decrease in the pH of the Earth's oceans 

An acropolis was the settlement of an upper part of an ancient Greek city, especially a citadel. 

Acuity may refer to:. 

Additive may refer to:. 

Adjudication is the legal process by which an arbiter or judge reviews evidence and argumentation. 

Admiralty most often refers to:Admiralty, Hong Kong 

Adrenaline, also known as epinephrine and alternatively spelled adrenalin, is a hormone and medication which is involved in regulating visceral functions.  

Aerospace refers to the technology and industry involved with the atmosphere and outer space collectively.  

Aesthetics is the branch of philosophy that studies beauty, taste, and related phenomena.  

Afforestation is the establishment of a forest or stand of trees in an area where there was no recent tree cover.  

In seismology, an aftershock is a smaller earthquake that follows a larger earthquake. 

Agility or nimbleness is an ability to change the body's position quickly. 

An agonist is a chemical that activates a receptor to produce a biological response.  

Agrarian means pertaining to agriculture, farmland, or rural areas. 

An airship, dirigible balloon or dirigible is a type of aerostat (lighter-than-air) aircraft that can navigate through the air flying under its own power.  

Albedo is the fraction of sunlight that is diffusely reflected by a body.  

Alchemy is an ancient branch of natural philosophy, a philosophical and protoscientific tradition. 

Algae is an informal umbrella term for any organisms from a large and diverse group of photosynthetic autotrophs/mixotrophs that are not plants. 

Algorithmic may refer to:Algorithm, step-by-step instructions for a calculation 

As a literary device or artistic form, an allegory is a narrative or visual representation in which a character, place, or event can be interpreted to represent a meaning with moral or political significance.  

An alliance is a relationship among people, groups, or states that have joined together for mutual benefit. 

Alluvium is loose clay, silt, sand, or gravel that has been deposited by running water in a stream bed, on a floodplain, in an alluvial fan or beach. 

An almanac is a regularly published listing of a set of current information about one or multiple subjects.  

Altruism is concern for the well-being, the life, of others independently of personal benefit or reciprocity. 

Amalgam most commonly refers to:Amalgam (chemistry), mercury alloy 

An ambassador is an official envoy, especially a high-ranking diplomat who represents a state. 

Ambiguity is a state in which the meaning of a phrase, statement, situation, or resolution is not explicitly defined. 

Ammonoids are extinct, typically coiled-shelled cephalopods composing the subclass Ammonoidea.  

Amnesty is defined as "A pardon extended by a government to a group or class of people". 

An amphitheatre is an open-air venue used for entertainment, performances, and sports.  

Anarchy is a form of society without rulers.  

Anemia is a blood disorder in which the blood has a reduced ability to carry oxygen.  

Flowering plants are plants that bear flowers and fruits, and form the clade Angiospermae.  

Anguish is "extreme unhappiness caused by physical or mental suffering." 

Anhydrite, or anhydrous calcium sulfate, is a mineral with the chemical formula CaSO4.  

In particle physics, annihilation is the process that occurs when a subatomic particle collides with its respective antiparticle to produce other particles. 

Annulus or annular indicates a ring- or donut-shaped area or structure.  

Anorexia nervosa (AN), often referred to simply as anorexia, is an eating disorder characterized by predominant food restriction. 

The Antarctic is the polar region of Earth that surrounds the South Pole, lying within the Antarctic Circle.  

Antiquity or Antiquities may refer to:. 

The aorta is the main and largest artery in the human body, originating from the left ventricle of the heart. 

Aphasia, also known as dysphasia, is an impairment in a person's ability to comprehend or formulate language because of dysfunction in specific brain regions.  

Aphids are small sap-sucking insects in the family Aphididae.  

An apsis is the farthest or nearest point in the orbit of a planetary body about its primary body.  

Apothecary is an archaic English term for a medical professional who formulates and dispenses materia medicacode: lat promoted to code: la  ('medicine') to physicians, surgeons and patients.  

Apparatus may refer to:Technical term for a body of the Soviet and post-Soviet governments 

Appeasement, in an international context, is a diplomatic negotiation policy of making political, material, or territorial concessions to an aggressive power with intention to avoid conflict.  

Appendix may refer to:. 

Apprenticeship is a system for training potential new practitioners of a trade or profession with on-the-job training and often some accompanying study.  

An aquifer is an underground layer of water-bearing material consisting of permeable or fractured rock, or of unconsolidated materials.  

Arable relates to the growing of crops:Arable farming or agronomy, the cultivation of field crops 

An arbiter or arbitrator is a person by whose decision the parties to a dispute agree to be bound in arbitration.  

Arboreal locomotion is the locomotion of animals in trees.  

Arcade most often refers to:Arcade game, a coin-operated video, pinball, electro-mechanical, redemption, etc., game 

The concept of an archetype appears in areas relating to behavior, historical psychology, philosophy and literary analysis. 

Ardor or Ardour may refer to:Ardor (album), a 1994 album by Love Spirals Downwards 

A cant is the jargon or language of a group, often employed to exclude or mislead people outside the group.  

Aridity is the condition of geographical regions which make up approximately 43% of total global available land area. 

Aristocracy is a form of government that places power in the hands of a small, privileged ruling class, the aristocrats. 

An armistice is a formal agreement of warring parties to stop fighting.  

Aromatherapy is a practice based on the use of aromatic materials, including essential oils and other aroma compounds. 

Arsenic is a chemical element; it has the symbol As and atomic number 33.  

An artisan is a skilled craft worker who makes or creates material objects partly or entirely by hand.  

Ascent or The Ascent may refer to:. 

An ashram is a spiritual hermitage or a monastery in Hinduism. 

Asphalt most often refers to:Bitumen, also known as "liquid asphalt cement" or simply "asphalt", a viscous form of petroleum mainly used as a binder in asphalt concrete 

Aspiration or aspirations may refer to:. 

An assay is an investigative (analytic) procedure in laboratory medicine, mining, pharmacology, environmental biology and molecular biology. 

Asylum may refer to:. 

An atoll is a ring-shaped island, including a coral rim that encircles a lagoon.  

In physics, attenuation is the gradual loss of flux intensity through a medium.  

An auction is usually a process of buying and selling goods or services by offering them up for bids, taking bids, and then selling the item to the highest bidder. 

Auditory means of or relating to the process of hearing:Auditory system, the neurological structures and pathways of sound perception 

Augury was a Greco-Roman religious practice of observing the behavior of birds, to receive omens.  

An aureola or aureole is the radiance of luminous cloud which, in paintings of sacred personages, surrounds the whole figure. 

Autarky is the characteristic of self-sufficiency, usually applied to societies, communities, states, and their economic systems. 

An autoclave is a machine used to carry out industrial and scientific processes requiring elevated temperature and pressure in relation to ambient pressure and temperature.  

An autopsy is a surgical procedure that consists of a thorough examination of a corpse by dissection to determine the cause, mode, and manner of death. 

Auxins are a class of plant hormones with some morphogen-like characteristics.  

Greed is an insatiable desire for material gain or social value, such as status or power. 

Avulsion in general refers to a tearing away.  

Axial may refer to:one of the anatomical directions describing relationships in an animal body 

An azimuth is the horizontal angle from a cardinal direction, most commonly north, in a local or observer-centric spherical coordinate system. 

A bactericide or bacteriocide, sometimes abbreviated Bcidal, is a substance which kills bacteria.  

A bailiff is a manager, overseer or custodian – a legal officer to whom some degree of authority or jurisdiction is given.  

Bakelite, formally poly­oxy­benzyl­methylen­glycol­anhydride, is a thermosetting phenol formaldehyde resin. 

A balcony is a platform projecting from the wall of a building, supported by columns or console brackets. 

Ballast is dense material used as a weight to provide stability to a vehicle or structure.  

A banyan, also spelled banian, is a fig that develops accessory trunks from adjacent prop roots. 

Barbarism, barbarity, or barbarous may refer to:Barbarism (linguistics), a non-standard word, expression, or pronunciation 

A baritone is a type of classical male singing voice whose vocal range lies between the bass and the tenor voice-types.  

The Baroque is a Western style of architecture, music, dance, painting, sculpture, poetry, and other arts that flourished from the early 17th century until the 1750s.  

A barrister is a type of lawyer in common law jurisdictions that originated from the Inns of Court in the medieval English legal system.  

A basement is any floor of a building that is not above the grade plane.  

In Ancient Roman architecture, a basilica was a large public building with multiple functions that was typically built alongside the town's forum.  

A bastion is a structure projecting outward from the curtain wall of a fortification. 

Bauxite is a sedimentary rock with a relatively high aluminium content.  

A beacon is an intentionally conspicuous device designed to attract attention to a specific location.  

Beetles are insects that form the order Coleoptera, in the superorder Holometabola.  

The belfry is a structure enclosing bells for ringing as part of a building. 

A belief is a subjective attitude that something is true or a state of affairs is the case.  

The benthic zone is the ecological region at the lowest level of a body of water such as an ocean, lake, or stream. 

Beryl ( BERR-əl) is a mineral composed of beryllium aluminium silicate with the chemical formula Be3Al2(SiO3)6.  

Bifurcation or bifurcated may refer to:. 

In European militaries, a billet is a living-quarters to which a soldier is assigned to sleep.  

Biomechanics is the study of the structure, function and motion of the mechanical aspects of biological systems. 

A biopsy is a medical test commonly performed by a surgeon, an interventional radiologist, or an interventional cardiologist.  

Birth rate, also known as natality and crude birth rate, is the total number of live human births per 1,000 population for a given period divided by the length of the period in years.  

In church governance, a diocese or bishopric is the ecclesiastical district under the jurisdiction of a bishop. 

Bitumen is an immensely viscous constituent of petroleum.  

Blight is a specific symptom affecting plants in response to infection by a pathogenic organism. 

A blizzard is a severe snowstorm characterized by strong sustained winds and low visibility. 

A blockade is the act of actively preventing a country or region from receiving or sending out food, supplies, weapons, or communications. 

Bloom or blooming may refer to:. 

In Buddhism, a bodhisattva is a person who has attained, or is striving towards, bodhi or Buddhahood.  

Bole may refer to:. 

Bondage may refer to:. 

A borough is an administrative division in various English-speaking countries.  

A botnet is a group of Internet-connected devices, each of which runs one or more bots.  

In geology, a boulder is a rock fragment with size greater than 25.9 cm (10.2 in) in diameter.  

Bounty or bounties commonly refers to:Bounty (reward), an amount of money or other reward offered by an organization for a specific task done with a person or thing. 

Bovidae is the biological family of cloven-hoofed, ruminant mammals that includes cattle, bison, buffalo, antelopes, and goat-antelopes such as sheep and goats.  

In botany, a bract is a modified or specialized leaf, associated with a reproductive structure such as a flower, inflorescence axis or cone scale. 

Brass is an alloy of copper and zinc, in proportions which can be varied to achieve different colours and mechanical, electrical, acoustic, and chemical properties. 

Breach, Breached, or The Breach may refer to:. 

Brevity is concision or brevitas, the quality of being brief or concise. 

A brigade is a major tactical military formation that typically comprises three to six battalions plus supporting elements.  

Brine is a high-concentration solution of salt in water.  

A bristle is a stiff hair or feather, either on an animal, such as a pig, a plant, or on a tool such as a brush or broom. 

A bronchus is a passage or airway in the lower respiratory tract that conducts air into the lungs.  

Bronze is an alloy consisting primarily of copper, commonly with about 12–12.5% tin and often with the addition of other metals. 

Brotherhood or The Brotherhood may refer to:. 

Bureau may refer to:. 

Burgundy is a historical region in France, encompassing the territory of the former administrative region of the same name. 

Burial, also known as interment or inhumation, is a method of final disposition whereby a dead body is placed into the ground. 

Hessian, burlap in North America, or crocus in The Caribbean, is a woven fabric made of vegetable fibres. 

A bushel is an imperial and US customary unit of volume, based upon an earlier measure of dry capacity.  

In geomorphology, a butte is an isolated hill with steep, often vertical sides and a small, relatively flat top. 

A by-law, is a set of rules or law established by an organization or community so as to regulate itself. 

A cabal is a group of people who are united in some close design, usually to promote their private views or interests in an ideology, a state, or another community. 

Cache, caching, or caché may refer to:. 

In materials science, a refractory is a material that is resistant to decomposition by heat or chemical attack. 

Boat racing (Regatta) is a sport in which boats, or other types of watercraft, race on water.  

In a monarchy, a regent is a person appointed to execute the office of a monarch temporarily.  

In politics, a regime is a system of government that determines access to public office. 

The reindeer or caribou is a species of deer with circumpolar distribution. 

A reliquary is a container for relics.  

Remnant or remnants may refer to:. 

Reptiles, as commonly defined, are tetrapod vertebrate animals with an ectothermic metabolism and amniotic development.  

Residue may refer to:. 

A resin is a solid or highly viscous liquid that can be converted into a polymer.  

A reticle or reticule, also known as a graticule or crosshair, is a pattern of fine lines or markings built into the eyepiece of an optical device. 

A retinue is a body of persons "retained" in the service of a noble, royal personage, or dignitary; a suite of retainers. 

Revelry may refer to:The revelries of Saturnalia 

Reverie may refer to:A daydream or a dreamy state. 

Rhapsody may refer to:. 

Rhenium is a chemical element; it has symbol Re and atomic number 75.  

Rhododendron, from Ancient Greek ῥόδον (rhódon), meaning "rose", and δένδρον (déndron), meaning "tree", is a very large genus of about 1,024 species of woody plants in the heath family (Ericaceae).  

Rhodonite is a manganese inosilicate, with the formula (Mn, Fe, Mg, Ca)SiO3, and member of the pyroxenoid group of minerals. 

Rhubarb is the fleshy, edible stalks (petioles) of species and hybrids of Rheum in the family Polygonaceae. 

The Rialto is a central area of Venice, Italy, in the sestiere of San Polo.  

A ribosome is a ribonucleoprotein particle found in all cells, both prokaryotic and eukaryotic, responsible for the synthesis of proteins.  

A ridge is a long, narrow, elevated geomorphologic landform, structural feature, or a combination of both separated from the surrounding terrain by steep sides.  

Rigging comprises the system of ropes, cables and chains, which support and control a sailing ship or sail boat's masts and sails.  

In hillslope geomorphology, a rill is a shallow channel cut into soil by the erosive action of flowing surface water.  

Rime may refer to:Rime ice, ice that forms when water droplets in fog freeze to the outer surfaces of objects, such as trees. 

A stream is a continuous body of surface water flowing within the bed and banks of a channel.  

A road is a thoroughfare from one place to another, primarily used for movement of traffic.  

An easel is an upright support used for displaying and/or fixing something resting upon it. 

Echelon may refer to:. 

Effluent is wastewater from sewers or industrial outfalls that flows directly into surface waters. 

Egress may refer to:Ingress, egress, and regress, legal terms referring to an individual's right to travel or move 

Sambucus is a genus of between 20 and 30 species of flowering plants in the family Adoxaceae.  

Embankment may refer to:. 

Emissary may refer to:. 

Enamel may refer to:. 

An enclave is a territory that is entirely surrounded by the territory of only one other state or entity.  

The endosperm is a tissue produced inside the seeds of most of the flowering plants following double fertilization.  

Engram may refer to:Engram (neuropsychology), a physical means by which memory traces are stored 

An entourage is an informal group or band of people who are closely associated with a (usually) famous, notorious, or otherwise notable individual.  

Epaulette is a type of ornamental shoulder piece or decoration used as insignia of rank by armed forces and other organizations.  

An epitaph is a short text honoring a deceased person.  

An equerry is an officer of honour.  

Ermine may refer to three species of mustelid in the genus Mustela, or their fur:Stoat or Eurasian ermine, Mustela erminea. 

An escarpment is a steep slope or long cliff that forms as a result of faulting or erosion and separates two relatively level areas having different elevations. 

An esplanade or promenade is a long, open, level area, usually next to a river or large body of water, where people may walk.  

Abstract may refer to:"Abstract", a 2017 episode of the animated television series Adventure Time 

Absolution is a theological term for the forgiveness imparted by ordained Christian priests and experienced by Christian penitents.  

Accord may refer to:. 

Acrimony may refer to:a feeling of hatred 

An adversary is generally considered to be a person, group, or force that opposes and/or attacks. 

An advocate is a professional in the field of law.  

Wealth is the abundance of valuable financial assets or physical possessions which can be converted into a form that can be used for transactions.  

Ambivalent may refer to:Ambivalence, a state of conflicting beliefs or feelings 

Analogy is a comparison or correspondence between two things because of a third element that they are considered to share.  

An antagonist is a character in a story who is presented as the main enemy or rival of the protagonist and is often depicted as a villain. 

Apathy, also referred to as indifference, is a lack of feeling, emotion, interest, and/or concern about something.  

Apex may refer to:. 

Arbitrariness is the quality of being "determined by chance, whim, or impulse, and not by necessity, reason, or principle".  

Archaic may refer to:Archaic Period, archaeological term used to refer to a very early period differing by location 

Articulate may refer to:Articulate!, a board game in which players describe words from different categories 

Assertion or assert may refer to:. 

Assimilate is a 2019 American science fiction horror film directed by John Murlowski and starring Joel Courtney, Andi Matichak, and Calum Worthy also with Mason McNulty and Cam Gigandet. 

Astute may refer to:HMS Astute (P447), launched 1945, Amphion-class submarine, scrapped 1970 

An axiom, postulate, or assumption is a statement that is taken to be true, to serve as a premise or starting point for further reasoning and arguments.  

Banal may refer to:Of or pertaining to the ban (medieval) or banalitéBanal nationalism. 

Benevolence or Benevolent may refer to:Benevolent (band) 

Bias is a disproportionate weight in favor of or against an idea or thing, usually in a way that is inaccurate, closed-minded, prejudicial, or unfair.  

A bolster is a long narrow pillow or cushion filled with cotton, down or fibre.  

Candid may refer to:Candid (app), a mobile app for anonymous discussions 

Capricious may refer to:Capricieuse, also spelled Capricious, a solitaire card game 

A censure is an expression of strong disapproval or harsh criticism.  

Cerebral may refer to:Of or relating to the brain 

Chronic may refer to:Chronic condition, a condition or disease that is persistent or otherwise long-lasting in its effects 

Coercion involves compelling a party to act in an involuntary manner through the use of threats, including threats to use force against that party.  

Cohesion may refer to:Cohesion (chemistry), the intermolecular attraction between like-molecules 

Colloquialism is the linguistic style used for casual (informal) communication.  

Concession may refer to:. 

In common usage and linguistics, concision is a communication principle of eliminating redundancy. 

In mathematics, a conjecture is a proposition that is proffered on a tentative basis without proof.  

A connotation is a commonly understood cultural or emotional association that any given word or phrase carries, in addition to its explicit or literal meaning, which is its denotation. 

Contingency or Contingent may refer to:Contingency (philosophy), in philosophy and logic 

Convolute may refer to:Convolute (botany) 

Credibility comprises the objective and subjective components of the believability of a source or message.  

Criterion may refer to:. 

In observational astronomy, culmination is the passage of a celestial object across the observer's local meridian.  

Cynicism is an attitude characterized by a general distrust of the motives of others.  

Debacle may refer to:an event that turns out to be a disaster 

Debate is a process that involves formal discourse, discussion, and oral addresses on a particular collection of topics. 

A debunker is a person or organization that exposes or discredits claims believed to be false, exaggerated, or pretentious.  

A decree is a legal proclamation, usually issued by a head of state, judge, royal figure, or other relevant authorities. 

Deduction may refer to:. 

A deficit is the amount by which a sum falls short of some reference amount. 

Deference is the condition of submitting to the espoused, legitimate influence of one's superior or superiors.  

Demise is an Anglo-Norman legal term for the transfer of an estate, especially by lease.  

In philosophy and linguistics, the denotation of a word or expression is its strictly literal meaning.  

Derive may refer to:Derive, a commercial system made by Texas Instruments 

Desolation or Desolate may refer to:Loneliness, an unpleasant emotional response to perceived isolation. 

detriment may refer to:detriment (astrology) 

A dichotomy is a partition of a whole into two parts (subsets).  

Diligence—carefulness and persistent effort or work—is listed as one of the seven capital virtues.  

Discreet may refer to:Discreet Logic, a subsidiary of Autodesk Media and Entertainment 

Altercation — a noisy or heated argument. 

Amalgamate — to combine or unite into one entity. 

Amass — to gather or accumulate a large amount. 

Ambit — the scope or extent of something. 

Ameliorate — to improve or make a situation better. 

Amicable — friendly and peaceful in tone or behavior. 

Amorphous — lacking a clear shape or structure. 

Anachronism — something out of place in its time period. 

Anathema — something or someone intensely disliked. 

Animosity — strong hostility or resentment. 

Annul — to declare something legally invalid. 

Anodyne — harmless or unlikely to offend; soothing. 

Antedate — to occur earlier than something else. 

Anthology — a collection of literary works. 

Antipathy — a deep feeling of dislike. 

Antiquated — outdated or no longer useful. 

Apathetic — showing little interest or emotion. 

Apostate — someone who abandons a belief or cause. 

Apparition — a ghostly or supernatural appearance. 

Apposite — highly relevant or appropriate. 

Apprehend — to arrest someone or understand something. 

Arbiter — a person with authority to settle disputes. 

Arboreal — relating to or living in trees. 

Archetype — the original model from which others are copied. 

Arduous — requiring great effort or endurance. 

Arid — extremely dry or lacking interest. 

Arrogate — to claim something without justification. 

Artifice — clever trickery or deception. 

Ascetic — practicing strict self‑discipline and simplicity. 

Ascribe — to attribute something to a cause or source. 

Askew — not straight or aligned; crooked. 

Aspersion — a damaging or critical remark. 

Assail — to attack violently or criticize strongly. 

Assiduous — showing great care and persistence. 

Assuage — to relieve or lessen an unpleasant feeling. 

Atavistic — relating to ancestral or primitive traits. 

Attenuate — to weaken or reduce in force. 

Audacious — boldly daring or disrespectfully bold. 

Augment — to increase or make larger. 

Auspicious — indicating a favorable or successful outcome. 

Austere — strict, plain, or severely simple. 

Autonomy — the right or condition of self‑governance. 

Avarice — extreme greed for wealth. 

Avert — to prevent something or turn away. 

Aversion — a strong dislike or unwillingness. 

Avow — to openly declare or admit something. 

Barrage — a concentrated, overwhelming outpouring. 

Bastion — a stronghold or protected position. 

Beleaguer — to trouble or harass persistently. 

Belie — to give a false impression of something. 

Bellicose — eager to fight or aggressively hostile. 

Belligerent — hostile, aggressive, or ready to fight. 

Benign — gentle, harmless, or not dangerous. 

Bequeath — to leave something to someone in a will. 

Berate — to scold or criticize harshly. 

Beseech — to beg urgently or earnestly. 

Bestow — to give something as an honor or gift. 

Bifurcate — to divide into two branches. 

Blatant — obvious and offensive in behavior. 

Blithe — carefree or showing a lack of concern. 

Bombastic — overly inflated or pretentious in speech. 

Boon — a helpful or beneficial thing. 

Burgeon — to grow or expand rapidly. 

Buttress — to support or strengthen something. 

Cacophony — a harsh, jarring mixture of sounds. 

Cadence — the rhythmic flow of sounds or movement. 

Cajole — to persuade through flattery or coaxing. 

Callous — emotionally insensitive or unfeeling. 

Candor — honest and straightforward expression. 

Capitulate — to surrender or give in. 

Caricature — an exaggerated portrayal for comic effect. 

Castigate — to criticize severely. 

Cavalier — showing a dismissive or carefree attitude. 

Chagrin — distress caused by failure or humiliation. 

Chastise — to punish or scold harshly. 

Chicanery — trickery used to deceive. 

Chide — to mildly scold or rebuke. 

Choleric — easily angered or irritable. 

Circumspect — cautious and considering all risks. 

Clandestine — done secretly or covertly. 

Coalesce — to come together into one whole. 

Cogent — clear, logical, and convincing. 

Cognizant — aware or having knowledge of something. 

Coherent — logically connected and consistent. 

Collude — to secretly cooperate for deceitful purposes. 

Commiserate — to express sympathy or share sorrow. 

Compel — to force or strongly persuade. 

Complacency — self‑satisfaction that prevents awareness of danger. 

Comport — to behave in a particular manner. 

Concede — to admit something is true or valid. 

Conciliate — to calm anger or restore goodwill. 

Concur — to agree with someone or something. 

Condone — to accept or allow wrongdoing. 

Conflagration — a large, destructive fire. 

Conflate — to combine two things into one. 

Conformity — behavior that follows rules or norms. 

Confound — to confuse or surprise greatly. 

Congeal — to solidify or thicken. 

Congruent — in agreement or harmony. 

Conjectural — based on guesswork rather than proof. 

Connive — to secretly allow wrongdoing. 

Connoisseur — an expert judge of taste or quality. 

Conscript — to force someone into military service. 

Consecrate — to declare something sacred. 

Consolidate — to combine into a stronger whole. 

Consternation — sudden shock or dismay. 

Construe — to interpret or understand something. 

Consummate — showing high skill or completeness. 

Contemptuous — showing deep disrespect or scorn. 

Contend — to argue or struggle against something. 

Contention — a heated disagreement or claim. 

Contrite — feeling deep remorse or guilt. 

Contrition — sincere regret for wrongdoing. 

Conundrum — a difficult or puzzling problem. 

Convene — to gather or assemble. 

Convivial — friendly, lively, and enjoyable. 

Copious — abundant in quantity. 

Corollary — a natural consequence or result. 

Corporeal — relating to the physical body. 

Corroborate — to confirm or support evidence. 

Cosmopolitan — familiar with many cultures or places. 

Credence — belief or acceptance of something as true. 

Credulous — too ready to believe things. 

Culinary — related to cooking or food. 

Culpable — deserving blame or responsibility. 

Cursory — done quickly with little attention. 

Curtail — to reduce or limit something. 

Debase — to reduce the quality or value of something. 

Debilitate — to weaken or impair strength. 

Debunk — to expose the falseness of a claim. 

Decadent — characterized by moral or cultural decline. 

Decimate — to destroy a large portion of something. 

Decry — to publicly criticize or condemn. 

Defame — to damage someone’s reputation with false statements. 

Defer — to postpone or yield to another’s judgment. 

Defile — to make something dirty or impure. 

Defunct — no longer existing or functioning. 

Degrade — to treat someone with disrespect or reduce quality. 

Deign — to do something considered beneath one’s dignity. 

Deleterious — harmful or damaging. 

Demeanor — outward behavior or manner. 

Demur — to raise objections or show reluctance. 

Denigrate — to criticize unfairly or belittle. 

Denounce — to publicly condemn or accuse. 

Depict — to represent or describe in words or images. 

Deplete — to use up resources or supplies. 

Deplore — to strongly disapprove of something. 

Depravity — moral corruption or wickedness. 

Deprecate — to express disapproval of something. 

Derelict — abandoned or in poor condition. 

Deride — to mock or ridicule. 

Derision — contemptuous ridicule or mockery. 

Despotism — absolute power exercised cruelly. 

Destitute — lacking basic necessities. 

Desultory — lacking a plan or purpose. 

Detract — to reduce the value or importance of something. 

Deviant — departing from accepted norms. 

Dexterous — skillful and coordinated. 

Diaphanous — light, delicate, and translucent. 

Didactic — intended to teach or instruct. 

Diffident — shy or lacking self‑confidence. 

Dilapidated — in a state of disrepair. 

Dilatory — slow to act or causing delay. 

Dilemma — a difficult choice between two options. 

Diminutive — extremely small or tiny. 

Disavow — to deny responsibility or support. 

Disconcert — to unsettle or disturb. 

Disconsolate — deeply unhappy or without comfort. 

Discordant — lacking harmony or agreement. 

Discrepancy — a difference between two things that should match. 

Disdainful — showing contempt or scorn. 

Disgruntled — dissatisfied or irritated. 

Disheveled — messy or untidy in appearance. 

Disingenuous — insincere or pretending to know less. 

Disparage — to belittle or speak negatively about. 

Dispassionate — not influenced by emotion. 

Dispel — to drive away a feeling or belief. 

Disperse — to scatter or spread widely. 

Disquiet — a feeling of anxiety or unease. 

Dissenting — expressing disagreement with majority views. 

Dissident — someone who opposes official policy. 

Dissuade — to persuade someone not to do something. 

Distill — to extract the essential meaning. 

Dither — to hesitate or be indecisive. 

Divest — to strip away possessions or rights. 

Docile — easily taught or managed. 

Dogged — showing persistent determination. 

Dogmatic — asserting opinions as absolute truth. 

Dour — stern, gloomy, or unfriendly. 

Dubiety — doubt or uncertainty. 

Duplicity — deceitfulness or double‑dealing. 

Ebullient — full of cheerful energy. 

Eccentricity — odd or unusual behavior. 

Ecstatic — extremely happy or joyful. 

Edify — to instruct or improve morally. 

Efface — to erase or remove something. 

Effervescent — lively, bubbly, or enthusiastic. 

Egregious — outstandingly bad or shocking. 

Elated — extremely happy or proud. 

Elucidate — to make something clear or explain. 

Emancipate — to free from restraint or oppression. 

Embargo — an official ban on trade. 

Embitter — to make someone resentful. 

Embolden — to give someone courage or confidence. 

Embroil — to involve someone in conflict. 

Empathy — the ability to understand others’ feelings. 

Empirical — based on observation or experience. 

Emulate — to imitate in order to equal or surpass. 

Enamored — filled with love or admiration. 

Enervate — to weaken or drain energy. 

Engender — to cause or give rise to something. 

Engross — to fully occupy attention. 

Enmity — deep hostility or hatred. 

Enrapture — to fill with intense delight. 

Enshrine — to preserve something as sacred. 

Entreat — to beg or plead earnestly. 

Enumerate — to list items one by one. 

Envision — to imagine or picture something. 

Epiphany — a sudden realization or insight. 

Epitome — the perfect example of something. 

Equanimity — calmness under stress. 

Equivocal — ambiguous or open to multiple meanings. 

Eradicate — to completely eliminate something. 

Erroneous — incorrect or based on error. 

Erudite — having great knowledge or learning. 

Eschew — to deliberately avoid something. 

Ethereal — delicate, light, or otherworldly. 

Eulogy — a speech praising someone who has died. 

Euphoria — intense happiness or excitement. 

Evanescent — fading away quickly. 

Evince — to show clearly or make evident. 

Exalt — to praise highly or elevate. 

Exasperate — to irritate intensely. 

Excise — to cut out or remove. 

Exculpate — to clear from blame. 

Exemplary — serving as a desirable model. 

Exhort — to strongly encourage or urge. 

Exhume — to dig up something buried. 

Exigent — requiring immediate action. 

Exonerate — to free from blame or guilt. 

Expedite — to speed up a process. 

Explicit — stated clearly and in detail. 

Expunge — to erase or remove completely. 

Extemporaneous — spoken or done without preparation. 

Extol — to praise enthusiastically. 

Extraneous — irrelevant or unrelated. 

Extrapolate — to infer from known information. 

Extricate — to free from difficulty or entanglement. 

Facetious — treating serious issues with inappropriate humor. 

Fallacy — a mistaken belief based on faulty reasoning. 

Fastidious — very attentive to detail. 

Fathom — to understand something deeply. 

Fatuous — silly and pointless. 

Feasible — possible or practical to achieve. 

Felicity — great happiness or joy. 

Fervor — intense passion or enthusiasm. 

Fidelity — faithfulness or loyalty. 

Flagrant — obviously offensive or wrong. 

Flippant — not showing serious or respectful attitude. 

Florid — overly elaborate or excessively decorated. 

Fluctuate — to rise and fall irregularly. 

Foible — a minor weakness or flaw. 

Forbear — to refrain from doing something. 

Foreboding — a feeling that something bad will happen. 

Forego — to do without or give up. 

Foresight — the ability to anticipate the future. 

Forlorn — sad, abandoned, or hopeless. 

Fortitude — courage in facing hardship. 

Fortuitous — happening by chance, often luckily. 

Fractious — irritable and difficult to control. 

Frivolous — lacking seriousness or importance. 

The quick brown fox jumps over the lazy dog. 

Pack my box with five dozen liquor jugs. 

How vexingly quick daft zebras jump. 

The five boxing wizards jump quickly. 

Sphinx of black quartz judge my vow. 

Two driven jocks help fax my big quiz. 

Jackdaws love my big sphinx of quartz. 

One is the smallest positive integer. 

Two is the only even prime number. 

Three is the first odd prime number. 

Four is the square of two. 

Five is the number of senses humans have. 

Six is the number of sides on a cube. 

Seven is considered a lucky number. 

Eight is the number of legs on a spider. 

Nine is the square of three. 

Ten is the base of our number system. 

Eleven is the first two digit prime number. 

Twelve is the number of months in a year. 

Thirteen is often considered an unlucky number. 

Fourteen is two times seven. 

Fifteen is three times five. 

Sixteen is the square of four. 

Seventeen is a prime number. 

Eighteen is two times nine. 

Nineteen is the last teen number. 

Twenty is two times ten. 

Monday is the first day of the week. 

Tuesday is the second day of the week. 

Wednesday is the third day of the week. 

Thursday is the fourth day of the week. 

Friday is the fifth day of the week. 

Saturday is the sixth day of the week. 

Sunday is the seventh day of the week. 

January is the first month of the year. 

February is the second month of the year. 

March is the third month of the year. 

April is the fourth month of the year. 

May is the fifth month of the year. 

June is the sixth month of the year. 

July is the seventh month of the year. 

August is the eighth month of the year. 

September is the ninth month of the year. 

October is the tenth month of the year. 

November is the eleventh month of the year. 

December is the twelfth month of the year. 

Red is a primary color. 

Blue is a primary color. 

Yellow is a primary color. 

Green is made by mixing blue and yellow. 

Orange is made by mixing red and yellow. 

Purple is made by mixing red and blue. 

Black is the absence of light. 

White is all colors of light combined. 

Gray is between black and white. 

Brown is a mixture of several colors. 

North is a cardinal direction. 

South is a cardinal direction. 

East is a cardinal direction. 

West is a cardinal direction. 

Northeast is between north and east. 

Northwest is between north and west. 

Southeast is between south and east. 

Southwest is between south and west. 

Spring is a season of renewal and new growth. 

Summer is a season of warmth and long days. 

Autumn is a season of change and falling leaves. 

Winter is a season of cold and short days. 

Dog is a loyal companion and common pet. 

Cat is an independent and curious pet. 

Bird can fly through the air using wings. 

Fish lives underwater and breathes through gills. 

Horse is a large mammal used for riding. 

Cow provides milk and meat for humans. 

Pig is an intelligent and social animal. 

Sheep produces wool used for clothing. 

Chicken lays eggs and is kept on farms. 

Duck swims on water and can also fly. 

Abandon means to leave something behind completely and never return. 

Ability means the power or skill to do something successfully. 

Able means having the power or skill to accomplish a task. 

Abnormal means not normal or unusual in some significant way. 

Aboard means on or into a ship aircraft or other vehicle. 

Abort means to stop something before it is completed fully. 

About means on the subject of or concerning something specific. 

Above means at a higher level or position than something else. 

Abroad means in or to a foreign country or countries. 

Absence means the state of being away or not present. 

Absolute means complete or total without any limitation. 

Absorb means to take in or soak up liquid or information. 

Abstract means existing in thought but not having physical form. 

Abuse means to use something wrongly or treat badly. 

Academic means relating to education and scholarly learning. 

Accept means to receive willingly or agree to something. 

Access means the ability to enter or approach something. 

Accident means an unexpected event causing damage or injury. 

Accomplish means to achieve or complete something successfully. 

Account means a record of financial transactions. 

Accurate means correct in all details and exact. 

Achieve means to successfully reach a desired goal. 

Acid means a chemical substance with specific properties. 

Acknowledge means to accept or admit the existence of something. 

Acquire means to buy or obtain something for oneself. 

Across means from one side to the other. 

Act means to take action or do something specific. 

Action means the process of doing something. 

Active means engaging in activity or movement. 

Activity means a thing that is done for a purpose. 

Actor means a person who performs in plays or movies. 

Actual means existing in fact and not theoretical. 

Add means to join something to something else. 

Address means the details of where someone lives. 

Adjust means to change slightly for a better fit. 

Administer means to manage or supervise something. 

Admit means to confess or acknowledge something. 

Adopt means to take up or start using something. 

Adult means a fully grown person or animal. 

Advance means to move forward or make progress. 

Advantage means a condition giving greater opportunity. 

Adventure means an exciting or unusual experience. 

Advertise means to promote or publicize something. 

Advice means guidance or recommendations about what to do. 

Advise means to offer suggestions about the best course of action. 

Affair means a matter or situation of a particular kind. 

Affect means to influence or make a difference to something. 

Afford means to have enough money to pay for something. 

Afraid means feeling fear or anxiety about something. 

After means later in time or following something. 

Again means once more or another time. 

Against means in opposition to or touching something. 

Age means the length of time someone has lived. 

Agency means a business providing a particular service. 

Agent means a person who acts on behalf of another. 

Agree means to have the same opinion as someone else. 

Agreement means a negotiated arrangement between parties. 

Ahead means in front or further forward in time. 

Aid means help or assistance given to someone. 

Aim means to point or direct toward a target. 

Air means the invisible gas surrounding the earth. 

Aircraft means a vehicle that can fly in the air. 

Airport means a place where aircraft take off and land. 

Alarm means a warning signal or feeling of anxiety. 

Album means a collection of recordings or photographs. 

Alcohol means a chemical compound found in drinks. 

Alert means quick to notice things or a warning signal. 

Alien means belonging to a foreign country or extraterrestrial. 

Alike means similar to each other in appearance. 

Alive means living and not dead. 

All means the whole quantity or extent of something. 

Allow means to let someone do something. 

Almost means very nearly but not quite. 

Alone means having no one else present. 

Along means in a line or in company with someone. 

Already means before now or earlier than expected. 

Also means in addition or too. 

Alter means to change or make different. 

Alternative means one of two or more possibilities. 

Although means in spite of the fact that. 

Always means at all times or on every occasion. 

Amaze means to surprise greatly or astonish. 

Ambition means a strong desire to achieve something. 

Among means surrounded by or in the company of. 

Amount means a quantity of something. 

Amuse means to cause someone to laugh or smile. 

Analyze means to examine something in detail. 

Ancient means belonging to the very distant past. 

And means used to connect words or phrases. 

Angel means a spiritual being believed to act as a messenger. 

Anger means a strong feeling of annoyance or hostility. 

Angle means the space between two intersecting lines. 

Angry means feeling or showing strong annoyance. 

Animal means a living creature that moves and eats. 

Anniversary means the date on which something occurred. 

Announce means to make something known publicly. 

Annoy means to irritate or bother someone. 

Annual means occurring once every year. 

Another means one more person or thing. 

Answer means a response to a question. 

Anticipate means to expect or look forward to something. 

Anxiety means a feeling of worry or unease. 

Any means one or some of a thing. 

Anyone means any person at all. 

Anything means any object or matter. 

Anyway means in any case or regardless. 

Anywhere means in or to any place. 

Apart means separated by a distance. 

Apartment means a suite of rooms forming one residence. 

Apologize means to express regret for something done. 

Appear means to become visible or present. 

Appearance means the way someone or something looks. 

Apple means a round fruit with red or green skin. 

Apply means to make a formal request. 

Approach means to come near or nearer to something. 

Approve means to have a favorable opinion. 

Approximate means close to the actual but not exact. 

Area means a region or part of a space. 

Argue means to exchange diverging views angrily. 

Argument means an exchange of diverging views. 

Arise means to emerge or become apparent. 

Arm means the upper limb of the human body. 

Army means an organized military force. 

Around means on all sides or in various places. 

Arrange means to put in a particular order. 

Arrest means to seize by legal authority. 

Arrival means the action of arriving somewhere. 

Arrive means to reach a destination. 

Art means the expression of creative skill. 

Article means a piece of writing in a publication. 

Artist means a person who creates art. 

As means to the same degree or amount. 

Ash means the powdery residue left after burning. 

Ask means to say something to get information. 

Asleep means in a state of sleep. 

Aspect means a particular part of something. 

Assault means a violent physical attack. 

Assert means to state a fact confidently. 

Assess means to evaluate or estimate something. 

Asset means a useful or valuable thing. 

Assign means to allocate a task or resource. 

Assist means to help someone do something. 

Assume means to suppose something is true. 

Assure means to tell someone something positively. 

Astonish means to surprise or impress greatly. 

Astronaut means a person trained to travel in space. 

At means expressing location or arrival. 

Athlete means a person proficient in sports. 

Atmosphere means the envelope of gases surrounding the earth. 

Atom means the basic unit of a chemical element. 

Attach means to join or fasten something. 

Attack means to act aggressively against someone. 

Attempt means to make an effort to achieve something. 

Attend means to be present at an event. 

Attention means notice taken of someone or something. 

Attitude means a settled way of thinking. 

Attract means to draw interest toward something. 

Attribute means a quality regarded as characteristic. 

Auction means a public sale to the highest bidder. 

Audience means the assembled spectators at an event. 

Audio means relating to sound or its reproduction. 

Aunt means the sister of one parent. 

Author means the writer of a book or article. 

Authority means the power to give orders. 

Auto means a motor vehicle or automobile. 

Automatic means operating by itself without human input. 

Available means able to be used or obtained. 

Avenue means a broad road or pathway. 

Average means the typical or normal amount. 

Avoid means to keep away from something. 

Awake means not sleeping or conscious. 

Award means a prize given for achievement. 

Aware means having knowledge of something. 

Away means at a distance from a place. 

Awful means very bad or unxpleasant. 

Awkward means causing difficulty or embarrassment. 
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
