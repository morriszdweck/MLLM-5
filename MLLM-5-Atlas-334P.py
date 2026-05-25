"""
MLLM-5-ATLAS  —  Micro Language Model-5 (Advanced Tiny Language Architecture System)
====================================================================================

A major architectural upgrade from MLLM-5-Abyss, tuned for SPEED and ACCURACY:

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
Public Health (journal), published by Elsevier for the Royal Society for Public Health
Public Health a 2021 proposed comedy television series by Rob Tepper
Public Health, a May 22, 2014 episode of Debatten, a Norwegian television series
Public Health, a July 6, 2000 episode of Today's Environment, television series by Five Star Productions.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics.
Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.".
Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance.
Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce.
Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations.
Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them. According to multiple researchers, the term is currently being used to describe a broad spectrum of concepts, in a general sense used as "a synonym for international political relations", but more specifically "to imply the global structure of such relations"; this usage builds on an "early-twentieth-century term for a pseudoscience of political geography" and other pseudoscientific theories of historical and geographic determinism.
Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry.
Failed to fetch Energy Systems.
Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology.
Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics.
Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines. Automation has been achieved by various means including mechanical, hydraulic, pneumatic, electrical, electronic devices, and computers, usually in combination. Complicated systems, such as modern factories, airplanes, and ships typically use combinations of all of these techniques. The benefits of automation includes labor savings, reducing waste, savings in electricity costs, savings in material costs, and improvements to quality, accuracy, and precision.
Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries.
Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.".
Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers.
Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering.
Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work. Employees work in return for wages, which can be paid on the basis of an hourly rate, by piecework or an annual salary, depending on the type of work an employee does, the prevailing conditions of the sector and the bargaining power between the parties. Employees in some sectors may receive gratuities, bonus payments or stock options. In some types of employment, employees may receive benefits in addition to payment. Benefits may include health insurance, housing, and disability insurance.
Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena.
Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment. While sleep differs from wakefulness in terms of the ability to react to stimuli, it still involves active brain patterns, making it more reactive than a coma or disorders of consciousness.
A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations. Participation in hobbies encourages acquiring substantial skills and knowledge in that area. A list of hobbies changes with renewed interests and developing fashions, making it diverse and lengthy. Hobbies tend to follow trends in society. For example, stamp collecting was popular during the nineteenth and twentieth centuries as postal systems were the main means of communication; as of 2024, video games became more popular following technological advances. The advancing production, technology, and labour movements of the nineteenth century provided workers with more leisure time to engage in hobbies. Because of this, the efforts of people investing in hobbies has increased with time.
Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them. A typology of shopper types has been developed by scholars which identifies one group of shoppers as recreational shoppers, that is, those who enjoy shopping and view it as a leisure activity.
Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury.
Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community. Historically, most human societies use family as the primary purpose of attachment, nurturance, and socialization.
Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping. Leisure as an experience usually emphasizes dimensions of perceived freedom and choice. It is done for "its own sake", for the quality of experience and involvement. Other classic definitions include Thorstein Veblen's (1899) of "nonproductive consumption of time." Free time is not easy to define due to the multiplicity of approaches used to determine its essence. Different disciplines have definitions reflecting their common issues: for example, sociology on social forces and contexts and psychology as mental and emotional states and conditions. From a research perspective, these approaches have an advantage of being quantifiable and comparable over time and place.
A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged. In everyday US usage, however, "grocery store" is a synonym for supermarket, and is not used to refer to other types of stores that sell groceries. In the UK, shops that sell food are distinguished as grocers or grocery shops.
Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan.
Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment. Cleaning is often performed for aesthetic, hygienic, functional, safety, or environmental protection purposes. Cleaning occurs in many different contexts, and uses many different methods. Several occupations are devoted to cleaning.
Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well. Laundry has been part of history since humans began to wear clothes, so the methods by which different cultures have dealt with this universal human need are of interest to several branches of scholarship.
Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events.
Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies. These means of transmission may be divided into communication channels for multiplexing, allowing for a single medium to transmit several concurrent communication sessions. Long-distance technologies invented during the 19th, 20th and 21st centuries generally use electric power, and include the electrical telegraph, telephone, television, and radio.
Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks. Common features include:Online platforms enable users to create and share content and participate in social networking.
User-generated content—such as text posts or comments, digital photos or videos, and data generated through online interactions.
Service-specific profiles that are designed and maintained by the social media organization.
Social media helps the development of online social networks by connecting a user's profile with those of other individuals or groups.
Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission. Television is a mass medium for advertising, entertainment, news, and sports. The medium is capable of more than "radio broadcasting", which refers to an audio signal sent to radio receivers.
A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage.
A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations. Most wedding ceremonies involve an exchange of marriage vows by a couple; a presentation of a gift ; and a public proclamation of marriage by an authority figure or celebrant. Special wedding garments are often worn, and the ceremony is sometimes followed by a wedding reception. Music, poetry, prayers, or readings from religious texts or literature are also commonly incorporated into the ceremony, as well as superstitious customs.
A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances. Funerary customs comprise the complex of beliefs and practices used by a culture to remember and respect the dead, from interment, to various monuments, prayers, and rituals undertaken in their honour. Customs vary between cultures and religious groups. Funerals have both normative and legal components. Common secular motivations for funerals include mourning the deceased, celebrating their life, and offering support and sympathy to the bereaved; additionally, funerals may have religious aspects that are intended to help the soul of the deceased reach the afterlife, resurrection or reincarnation.
Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements—although there is no scholarly consensus over what precisely constitutes a religion. It is an essentially contested concept. Different religions may or may not contain various elements ranging from the divine, sacredness, faith, and a supernatural being or beings.
Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources.
The branch of social science that studies politics and government is referred to as political science.
History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past.
Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.".
Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine.
Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software. Technology plays a critical role in science, engineering, and everyday life.
Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty.
Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role.
Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions.
Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Economics is a social science that studies the production, distribution, and consumption of goods and services.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions.
A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. The fruit is variable in size, color and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a peel, which may have a variety of colors when ripe. It grows upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) cultivated bananas come from two wild species – Musa acuminata and Musa balbisiana, or their hybrids.
Orange most often refers to:Orange (fruit), the fruit of the tree species  Citrus × sinensis
Orange blossom, its fragrant flower
Orange juice
Orange (colour), the color of an orange fruit, occurs between red and yellow in the visible light spectrum
Some other citrus or citrus-like fruit, see list of plants known as orange
Orange (word), both a noun and an adjective in the English language.
The garden strawberry is a widely grown hybrid plant cultivated worldwide for its fruit. The genus Fragaria, the strawberries, is in the rose family, Rosaceae. The fruit is appreciated for its aroma, bright red colour, juicy texture, and sweetness. It is eaten either fresh or in prepared foods such as jam, ice cream, and chocolates. Artificial strawberry flavourings and aromas are widely used in commercial products. Botanically, the strawberry is not a berry, but an aggregate accessory fruit. Each apparent 'seed' on the outside of the strawberry is actually an achene, a botanical fruit with a seed inside it.
Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries. They are classified in the section Cyanococcus within the genus Vaccinium. Commercial blueberries—both wild (lowbush) and cultivated (highbush)—are all native to North America. The highbush varieties were introduced into Europe during the 1930s.
The raspberry is the edible fruit of several plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus. The name also applies to these plants themselves. Raspberries are perennial with woody stems.
The blackberry is an edible fruit produced by many species in the genus Rubus in the family Rosaceae, hybrids among these species within the subgenus Rubus, and hybrids between the subgenera Rubus and Idaeobatus. The taxonomy of blackberries has historically been confused because of hybridization and apomixis so that species have often been grouped together and called species aggregates.
The pineapple is a tropical plant with an edible fruit; it is the most economically significant plant in the family Bromeliaceae.
A mango is an edible stone fruit produced by the tropical tree Mangifera indica. It originated in the northeastern part of the Indian subcontinent, in what is now Bangladesh, northeastern India and Myanmar. M. indica has been cultivated in South and Southeast Asia since ancient times, resulting in two modern mango cultivar lineages: the "Indian" and the "Southeast Asian" types. Other species in the genus Mangifera also produce edible fruits called "mangoes," most of which are found in the Malesian ecoregion.
The papaya, papaw, or pawpaw is the plant species Carica papaya, one of the 21 accepted species in the genus Carica of the family Caricaceae. Papaya is also the name of its fruit. It was first domesticated in Mesoamerica, within modern-day southern Mexico and Central America. It is grown in several countries in regions with a tropical climate. In 2024, India was the leading producer, accounting for 36% of the world total.
A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes are a non-climacteric type of fruit, generally occurring in clusters.
The watermelon is a species of flowering plant in the family Cucurbitaceae, that has a large, edible fruit. It is a scrambling and trailing vine-like plant, and is widely cultivated worldwide, with more than 1,000 varieties.
The cantaloupe is a type of true melon with sweet, aromatic, and usually orange flesh. Originally, cantaloup referred to the true cantaloupe or European cantaloupe with non- to slightly netted and often ribbed rind. Today, it also refers to the muskmelon with strongly netted rind, which is called cantaloupe in North America, rockmelon in Australia and New Zealand, and spanspek in Southern Africa. Cantaloupes range in mass from 0.5 to 5 kilograms.
Honeydew may refer to:Honeydew (melon), a cultivar group of melon
Honeydew (secretion), a sugar-rich sticky substance secreted by various animals
Honeydew moth, a moth of Southern and Middle America
Honeydew, California, United States, a town
Honeydew, West Virginia, United States, an unincorporated community
Honeydew (color), a pale shade of the color spring green
Bunsen Honeydew, a fictional character from The Muppets franchise
Honeydew (album), a 2008 album by Shawn Mullins
Honeydew (film), a 2020 American horror film written and directed by Devereux Milburn
Honey Dew Donuts, a Massachusetts-based franchise selling donuts and other breakfast foods
Fuller's Organic Honey Dew, a brand of pale ale brewed by Fuller's Brewery
Simon "Honeydew" Lane, a member of internet gaming group The Yogscast
"Honeydew" , a 2023 episode of The Bear TV series

.
Kiwi most commonly refers to:Kiwi (bird), a flightless bird native to New Zealand
Kiwi (nickname), an informal name for New Zealanders
Kiwifruit, an edible hairy fruit with many seeds
Kiwi dollar or New Zealand dollar, a unit of currency.
The peach is a deciduous tree that bears edible juicy fruits with various characteristics. Most are simply called peaches, while the glossy-skinned, non-fuzzy varieties are called nectarines. Though from the same species, they are regarded commercially as different fruits.
A plum is a fruit of some species in Prunus subg. Prunus. Dried plums are usually called prunes.
A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe.
An apricot is a fruit, or the tree that bears the fruit, of several species in the genus Prunus. Usually an apricot is from the species Prunus armeniaca, but the fruits of the other species in Prunus sect. Armeniaca are also called apricots. In 2023, world production of apricots was 3.7 million tonnes, led by Turkey with 20% of the total.
The pomegranate is a fruit-bearing, deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows to between 1.5–5 metres (5–16 ft) tall. Rich in symbolic and mythological associations in many cultures, it originated from the Iranian plateau including Iran, the Caucasus, Turkmenistan, Afghanistan and Pakistan. Pomegranate was first domesticated by ancient Iranians in the Persian plateau and nearby regions about 5,000 years ago. It is extensively cultivated for its fruit.
The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers. It originated from western South America, and may have been domesticated there, in Mexico, or in Central America. The Spanish introduced tomatoes to Eurasia in the Columbian exchange in the 16th century.
The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables. Considered an annual plant, there are three main types of cucumber—slicing, pickling, and seedless—within which several cultivars have been created. The cucumber originates in Asia extending from India, Nepal, Bangladesh, China, and Northern Thailand, but now grows on most continents, and many different types of cucumber are grown commercially and traded on the global market. In North America, the term wild cucumber refers to plants in the genera Echinocystis and Marah, though the two are not closely related.
The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist, all of which are domesticated forms of the wild carrot, Daucus carota, native to Europe and Southwestern Asia. The plant probably originated in Iran and was originally cultivated for its leaves and seeds.
Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable. Broccoli is classified in the Italica cultivar group of the species Brassica oleracea. Broccoli has large flower heads, or florets, usually dark green, arranged in a tree-like structure branching out from a thick stalk, which is usually light green. Leaves surround the mass of flower heads. Broccoli resembles cauliflower, a different but closely related cultivar group of the same Brassica species.
Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family. Cauliflower usually grows with one main stem that carries a large, rounded "head" made of tightly clustered, immature white or off-white flower buds called the "curd". Typically, only the "head" is eaten.
Spinach is a leafy green flowering plant native to Central and Western Asia. It is of the order Caryophyllales, family Amaranthaceae, subfamily Chenopodioideae. Its leaves are a common vegetable consumed either fresh, cooked or after storage. The taste differs considerably between cooked and raw: the high oxalate content may be reduced by steaming.
Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant. Its multiple different cultivars vary quite a bit in appearance; the leaves can be bumpy, curly, or flat, and the color ranges from purple to green.
Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable. The leaves are most often used raw in green salads, although lettuce is also seen in other kinds of food, such as sandwiches, wraps and soups; it can also be grilled. Its stem and seeds are sometimes used; celtuce is one variety grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. In 2023, world production of lettuce was 28 million tonnes, led by China with 53% of the total.
Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula.
Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible. It is closely related, but not identical, to the marrow; its fruit may be called marrow when mature.
Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae. Solanum melongena is grown worldwide for its edible fruit, typically used as a vegetable in cooking.
The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers". While they are botanically fruits—classified as berries—they are commonly used as a vegetable ingredient or side dish. Other varieties of the genus Capsicum are categorized as chili peppers when they are cultivated for their pungency, including some varieties of Capsicum annuum.
The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium. The shallot is a botanical variety of the onion which was classified as a separate species until 2011. The onion's close relatives include garlic, scallion, leek, and chives.
Garlic is a species of bulbous flowering plants in the genus Allium. Its close relatives include the onion, shallot, leek, chives, Welsh onion, and Chinese onion. Garlic is native to central and western Asia, stretching from the Black Sea through the southern Caucasus, northeastern Iran, and the Hindu Kush. It has naturalized in many other parts of the world, including Mediterranean Europe and China. There are two subspecies and hundreds of varieties of garlic.
Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine. It is an herbaceous perennial that grows annual pseudostems about one meter tall, bearing narrow leaf blades. The inflorescences bear flowers having pale yellow petals with purple edges, and arise directly from the rhizome on separate shoots.
The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world. Potatoes are underground stem tubers of the plant Solanum tuberosum, a perennial in the nightshade family Solanaceae.
The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, which is a staple food in parts of the world. Cultivars of the sweet potato have been bred to bear tubers with flesh and skin of various colors. Moreover, the young shoots and leaves are occasionally eaten as greens. The sweet potato and the potato are only distantly related, both being in the order Solanales. Although darker sweet potatoes are often known as yams in parts of North America, they are even more distant from actual yams, which are monocots in the order Dioscoreales.
Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain. The leafy stalk of the plant gives rise to male inflorescences or tassels which produce pollen, and female inflorescences called ears. The ears yield grain, known as kernels or seeds. In modern commercial varieties, these are usually yellow or white; other varieties can be of many colors. Maize was domesticated by indigenous peoples in southern Mexico about 9,000 years ago from wild teosinte. Native Americans planted it alongside beans and squashes in the Three Sisters polyculture.
Could not find summary for "Green Beans".
Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia. Widely cultivated as a vegetable crop, its young shoots are used as a spring vegetable.
Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times.
A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground on soil or another food source. A toadstool generally refers to a poisonous mushroom.
The avocado, alligator pear or avocado pear is an evergreen tree in the laurel family (Lauraceae). It is native to the Americas, with archaeological evidence of early human avocado use dating back thousands of years across various regions of Central and South America. It was prized for its large and unusually oily fruit. The native range of avocado extends from Mexico to Peru, encompassing much of Central America and parts of northern and western South America.
Lime most commonly refers to:Lime (fruit), a green citrus fruit
Lime (material), inorganic materials containing calcium, usually calcium oxide or calcium hydroxide
Lime (color), a color between yellow and green.
The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae. A true lemon is a hybrid of the citron and the bitter orange. Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India. Some other citrus fruits are called lemon.
The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit. The flesh of the fruit is segmented and varies in color from pale yellow to dark red.
Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn. The pear tree and shrub are a species of genus Pyrus, in the family Rosaceae, bearing the pomaceous fruit of the same name. Several species of pears are valued for their edible fruit and juices, while others are cultivated as trees.
The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos. The term "coconut" can denote the whole coconut palm tree or the large hard fruit. Originally native to Central Indo-Pacific, they are ubiquitous in coastal tropical regions.
Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower. The fruit is a pepo, a type of botanical berry, round to oval, either yellow or dark purple at maturity, with a soft to firm, juicy interior filled with numerous seeds.
Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae.
The fruit is edible and has a sweet, mildly tart flavor and a distinctive floral aroma often described as rose-like.
The durian is the edible fruit of several tree species belonging to the genus Durio. There are 30 recognised species, at least nine of which produce edible fruit. Durio zibethinus, native to Borneo, Sumatra, and the Malay Peninsula, is the only species available on the international market. It has over 300 named varieties in Thailand and over 200 in Malaysia as of 2021. Other species are sold in their local regions.
Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions. The common guava Psidium guajava is a small tree in the myrtle family (Myrtaceae), native to Mexico, Central America, the Caribbean and northern South America.
Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia. The edible fruit has distinctive ridges running down its sides. When cut in cross-section, it resembles a star, giving it the name of star fruit. The entire fruit is edible, usually raw, and may be cooked or made into relishes, preserves, garnish, and juices. It is commonly consumed in Southeast Asia, South Asia, the South Pacific, Micronesia, parts of East Asia, the United States, parts of Latin America, and the Caribbean. The tree is cultivated throughout tropical areas of the world.
Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador. Pitaya is cultivated in East Asia, South Asia, Southeast Asia, continental America, the Caribbean, Australia, Brazil, Madeira (Portugal), and throughout tropical and subtropical regions of the world.
Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa. Rice is the seed of the grass species Oryza sativa —or, much less commonly, Oryza glaberrima. Asian rice was domesticated in China some 13,500 to 8,200 years ago; African rice was domesticated in Africa about 3,000 years ago. Rice has become commonplace in many cultures worldwide; in 2023, 800 million tons were produced, placing it third after sugarcane and maize. Only some 8% of rice is traded internationally. China, India, and Indonesia are the largest consumers of rice. A substantial amount of the rice produced in developing nations is lost after harvest through factors such as poor transport and storage. Rice yields can be reduced by pests including insects, rodents, and birds, as well as by weeds, and by diseases such as rice blast. Traditional rice polycultures such as rice-duck farming, and modern integrated pest management seek to control damage from pests in a sustainable way.
Pasta is a type of food typically made from an unleavened dough of wheat flour mixed with water or eggs, and formed into sheets or other shapes, then cooked by boiling or baking. Pasta was originally only made with durum, although the definition has been expanded to include alternatives for a gluten-free diet, such as rice flour, or legumes such as beans or lentils. Pasta is believed to have developed independently in Italy and is a staple food of Italian cuisine, with evidence of Etruscans making pasta as early as 400 BCE in Italy.
Bread is a baked food product made from water, flour, and often yeast. It is a staple food across the world, particularly in Europe and the Middle East. Throughout recorded history and around the world, it has been an important part of many cultures' diets. It is one of the oldest human-made foods, having been of significance since the dawn of agriculture, and plays an essential role in both religious rituals and secular culture.
A tortilla is a thin, circular unleavened flatbread from Mesoamerica originally made from masa, and now also from wheat flour.
The oat, sometimes called the common oat, is a species of cereal grass (Avena) grown for fodder and for its seed, which is known by the same name. Oats appear to have been domesticated as a secondary crop, as their seeds resembled those of other cereals closely enough for them to be included by early cultivators. Oats tolerate cold winters less well than cereals such as wheat, barley, and rye, but need less summer heat and more rain, making them important in areas such as Northwest Europe that have cool, wet summers. They can tolerate low-nutrient and acid soils. Oats grow thickly and vigorously, allowing them to outcompete many weeds, and compared to other cereals are relatively free from diseases.
Quinoa is a flowering plant in the amaranth family. It is a herbaceous annual plant grown as a crop primarily for its edible seeds; the seeds are high in protein, dietary fiber, B vitamins and dietary minerals especially potassium and magnesium in amounts greater than in many grains. Quinoa is not a grass but rather a pseudocereal botanically related to spinach and amaranth, and originated in the Andean region of northwestern South America. It was first used to feed livestock 5,200–7,000 years ago, and for human consumption 3,000–4,000 years ago in the Lake Titicaca basin of Bolivia and Peru.
Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally. One of the first cultivated grains, it was domesticated in the Fertile Crescent around 9000 BC, giving it nonshattering spikelets and making it much easier to harvest. Its use then spread throughout Eurasia by 2000 BC. Barley prefers relatively low temperatures and well-drained soil to grow. It is relatively tolerant of drought and soil salinity, but is less winter-hardy than wheat or rye.
The lentil is an annual legume grown for its lens-shaped edible seeds or pulses, also called lentils. It is about 40 cm (16 in) tall, and the seeds grow in pods, usually with two seeds in each.
The chickpea or chick pea is an annual legume of the family Fabaceae, subfamily Faboideae, cultivated for its edible seeds. Its different types are variously known as gram, Bengal gram, chana dal, garbanzo, garbanzo bean, or Egyptian pea. It is one of the earliest cultivated legumes, the oldest archaeological evidence of which was found in Syria.
Could not find summary for "Black Beans".
The kidney bean is a variety of the common bean ; it has such a common name owing to its resemblance to a human kidney.
Tofu  or bean curd is a food prepared by pressing the curds of coagulated soy milk into solid white blocks of varying softness: silken, soft, firm, and extra firm.
Tempeh or tempe is a traditional Indonesian food made from fermented soybeans. It is made by a natural culturing and controlled fermentation process that binds soybeans into a cake form. A fungus, Rhizopus oligosporus or Rhizopus oryzae, is used in the fermentation process and is also known as tempeh starter.
The chicken is a domesticated form of the red junglefowl, originally native to Southeast Asia. It was first domesticated around 8,000 years ago and is one of the most common and widespread domesticated animals in the world. Chickens are primarily kept for their meat and eggs, though they are also kept as pets.
Beef is the culinary name for meat from cattle. Beef can be prepared in various ways; cuts are often used for steak, which can be cooked to varying degrees of doneness, while trimmings are often ground or minced, as found in most hamburgers. Beef contains protein, iron, and vitamin B12. Along with other kinds of red meat, high consumption is associated with an increased risk of colorectal cancer and cardiovascular disease, especially when processed. Beef has a high environmental impact, being a primary driver of deforestation with the highest greenhouse gas emissions of any agricultural product.
Pork is the culinary name for the meat of the pig. It is the second most commonly consumed type of meat worldwide, following poultry, with evidence of pig husbandry dating back to 8000–9000 BCE.
Turkey, officially the Republic of Türkiye, is a country mainly located in Anatolia in West Asia, with a smaller part called East Thrace in Southeast Europe. It borders the Black Sea to the north; Georgia, Armenia, Azerbaijan, and Iran to the east; Iraq, Syria, and the Mediterranean Sea to the south; and the Aegean Sea, Greece, and Bulgaria to the west. Turkey is home to over 86 million people; most are ethnic Turks, while Kurds are the largest ethnic minority. Officially a secular state, Turkey has a Muslim-majority population. Ankara is Turkey's capital and second-largest city. Istanbul is its largest city and economic center. Other major cities include İzmir, Bursa, and Antalya.
Salmon are any of several commercially important species of euryhaline ray-finned fish from the genera Salmo and Oncorhynchus of the family Salmonidae, native to tributaries of the North Atlantic (Salmo) and North Pacific (Oncorhynchus) basins. Salmon is a colloquial or common name used for fish in this group, but is not a scientific name. Other closely related fish in the same family include trout, char, grayling, whitefish, lenok and taimen, all coldwater fish of the subarctic and cooler temperate regions with some sporadic endorheic populations in Central Asia.
A tuna is a saltwater fish that belongs to the tribe Thunnini, a subgrouping of the Scombridae (mackerel) family. The Thunnini comprise 15 species across five genera, the sizes of which vary greatly, ranging from the bullet tuna up to the Atlantic bluefin tuna, which averages 2 m (6.6 ft) and is believed to live up to 50 years.
A shrimp is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion – usually decapods belonging to the Caridea or Dendrobranchiata, although some crustaceans outside of this order are also referred to as "shrimp".
Crabs are decapod crustaceans, either the Brachyura or various groups within the closely related Anomura, characterised by having a heavily armoured shell, their tail segments concealed under the body, the ability to run sideways, and the habit of hiding in rocky crevices. They do not form a single natural group or clade, but have convergently evolved multiple times from the ancestral decapod body plan through carcinisation, the process of creating this set of characteristics. As a group, they are thus polyphyletic, meaning they have multiple evolutionary origins.
Lobsters are malacostracan decapod crustaceans of the family Nephropidae or its synonym Homaridae. They have long bodies with muscular tails and live in crevices or burrows on the sea floor. Three of their five pairs of legs have claws, including the first pair, which are usually much larger than the others. Highly prized as seafood, lobsters are economically important and are often one of the most profitable commodities in the coastal areas they populate.
An egg is an organic vessel in which an embryo begins to develop.
Milk is a usually white liquid food produced by the mammary glands of lactating mammals. It is the primary source of nutrition for young mammals before they are able to digest solid food. Milk contains many nutrients, including calcium and protein, as well as lactose and saturated fat; the enzyme lactase is needed to break down lactose. Immune factors and immune-modulating components in milk contribute to milk immunity. The first milk, which is called colostrum, contains antibodies and immune-modulating components that strengthen the immune system against many diseases.
Cheddar cheese is a natural cheese that is relatively hard, off-white, and sometimes sharp-tasting. It originates from the village of Cheddar in Somerset, South West England.
Mozzarella is a semi-soft non-aged cheese prepared using the pasta filata ('stretched-curd') method. It originated in southern Italy.
Yogurt is a food produced by bacterial fermentation of milk. Fermentation of sugars in the milk by these bacteria produces lactic acid, which acts on milk protein to give yogurt its texture and characteristic tart flavor. Cow's milk is most commonly used to make yogurt. Milk from water buffalo, goats, ewes, mares, camels, and yaks is also used to produce yogurt. The milk used may be homogenized or not. It may be pasteurized or raw. Each type of milk produces substantially different results.
Butter is a dairy product made from the fat and protein components of churned cream. It is a semi-solid emulsion at room temperature, consisting of approximately 81% butterfat. It is used at room temperature as a spread, melted as a condiment, and used as a fat in baking, sauce-making, pan frying, and other cooking procedures.
The almond is a species of tree from the genus Prunus. Along with the peach, it is classified in the subgenus Amygdalus, distinguished from the other subgenera by corrugations on the shell (endocarp) surrounding the seed.
A walnut is the edible seed of any tree of the genus Juglans, particularly the Persian or English walnut, Juglans regia. They are accessory fruit because the outer covering of the fruit is technically an involucre and thus not morphologically part of the carpel; this means it cannot be a drupe but is instead a drupe-like nut.
Cashew is the common name of a tropical evergreen tree Anacardium occidentale, in the family Anacardiaceae. It is the source of the cashew nut and the cashew apple. The tree can grow as tall as 14 meters.
Peanuts is a syndicated daily and Sunday American comic strip written and illustrated by Charles M. Schulz. The strip originally ran from 1950 to 2000, continuing in reruns afterward. Peanuts is regarded as one of the most popular and influential comic strips in history, with 17,897 strips published in all, making it "arguably the longest story ever told by one human being". At the time of Schulz's death in 2000, Peanuts ran in over 2,600 newspapers, with a readership of roughly 355 million across 75 countries, and had been translated into 21 languages. It helped to cement the four-panel gag strip as the standard in the United States, and together with its merchandise earned Schulz more than $1 billion. Following successful animated television and stage-theatrical adaptations over the years, five animated theatrical films have been released.
Sunflower seeds are the seeds of the sunflower (Helianthus).
Could not find summary for "Pumpkin Seeds".
Olive oil is a vegetable oil obtained by pressing whole olives and extracting the oil.
Honey is a sweet and viscous substance made by several species of bees, the best-known of which are honey bees. Honey is made and stored to nourish bee colonies. Bees produce honey by gathering and then refining the sugary secretions of plants or the secretions of other insects, like the honeydew of aphids. This refinement takes place both within individual bees, through regurgitation and enzymatic activity, and during storage in the hive, through water evaporation that concentrates the honey's sugars until it is thick and viscous.
Maple syrup is a sweet syrup made from the sap of maple trees. In cold climates these trees store starch in their trunks and roots before winter; the starch is then converted to sugar that rises in the sap in late winter and early spring. Maple trees are tapped by drilling holes into their trunks and collecting the sap, which is heated to evaporate much of the water, leaving the concentrated syrup.
Chocolate is a food made from roasted and ground cocoa beans that can be a liquid, solid, or paste, either by itself or to flavor other foods. Cocoa beans are the processed seeds of the cacao tree. They are usually fermented to develop the flavor, then dried, cleaned, and roasted. The shell is removed to reveal nibs, which are ground to chocolate liquor The liquor can be processed to separate its two components, cocoa solids and cocoa butter, or shaped and sold as unsweetened baking chocolate. By adding sugar, sweetened chocolates are produced, which can be sold simply as dark chocolate, or, with the addition of milk, can be made into milk chocolate. Making milk chocolate with cocoa butter and without cocoa solids produces white chocolate.
Vanilla is a spice derived from orchids of the genus Vanilla, primarily obtained from the seed pods of the flat-leaved New World vanilla (V. planifolia).
Cinnamon is a spice obtained from the inner bark of several tree species from the genus Cinnamomum. Cinnamon is used mainly as an aromatic condiment and flavouring additive in a wide variety of cuisines, in particular sweet and savoury dishes such as biscuits, breakfast cereals, snack foods, bagels, teas, hot chocolate, and traditional foods. The aroma and flavour of cinnamon derive from its essential oil and principal component, cinnamaldehyde, as well as numerous other constituents, including eugenol.
Basil, also called great basil, is a culinary herb of the family Lamiaceae (mints). It is a tender plant, and is used in cuisines worldwide. In Western cuisine, the generic term "basil" refers to the variety also known as Genovese basil or sweet basil. Basil is native to tropical regions from Central Africa to Southeast Asia. In temperate climates basil is treated as an annual plant, but it can be grown as a short-lived perennial or biennial in warmer horticultural zones with tropical or Mediterranean climates.
Oregano is a species of flowering plant in the mint family, Lamiaceae. It was native to the Mediterranean region, but widely naturalised elsewhere in the temperate Northern Hemisphere.
Parsley, or garden parsley, is a species of flowering plant in the family Apiaceae that is native to the Balkans. It has been introduced and naturalized in Europe and elsewhere in the world with suitable climates, and is widely cultivated as a herb and a vegetable.
Mint or The Mint may refer to:.
Salvia rosmarinus, synonym Rosmarinus officinalis, commonly known as rosemary, is a shrub with fragrant, evergreen, needle-like leaves and purple or sometimes white, pink, or blue flowers. It is a member of the mint family, Lamiaceae.
Thyme is a culinary herb consisting of the dried aerial parts of some members of the genus Thymus of flowering plants in the mint family Lamiaceae. Thymes are native to Eurasia and north Africa. Thymes have culinary, medicinal, and ornamental uses. The species most commonly cultivated and used for culinary purposes is Thymus vulgaris, native to Southeast Europe.
A telephone, commonly shortened to phone, is a telecommunications device that enables two or more users to conduct a conversation when they are too far apart to be easily heard directly. A telephone converts sound, typically and most efficiently the human voice, into electronic signals that are transmitted via cables and other communication channels to another telephone which reproduces the sound to the receiving user. The term is derived from Ancient Greek: τῆλε, romanized: tēle, lit. 'far' and φωνή, together meaning distant voice.
A laptop is a portable personal computer (PC). Laptops typically have a clamshell form factor with a flat-panel screen on the inside of the upper lid and an alphanumeric keyboard and pointing device on the inside of the lower lid. Most of the computer's internal hardware is in the lower part, under the keyboard, although many modern laptops have a built-in webcam at the top of the screen, and some even feature a touchscreen display. In most cases, unlike tablet computers which run on mobile operating systems, laptops tend to run on desktop operating systems, which were originally developed for desktop computers.
Tablet may refer to:.
Keyboard may refer to:.
A mouse is a small rodent. Characteristically, mice are known to have a pointed snout, small rounded ears, a body-length scaly tail, and a high breeding rate. The best known mouse species is the common house mouse. Mice are also popular as pets. In some places, certain kinds of field mice are locally common. They are known to invade homes for food and shelter.
Monitor or monitor may refer to:.
Headphones are a pair of small loudspeaker drivers worn on or around the head over a user's ears. They are electroacoustic transducers, which convert an electrical signal to a corresponding sound. Headphones let a single user listen to an audio source privately, in contrast to a loudspeaker, which emits sound into the open air for anyone nearby to hear. Headphones are also known as earphones or, colloquially, cans. Circumaural and supra-aural headphones use a band over the top of the head to hold the drivers in place. Another type, known as earbuds or earpieces, consists of individual units that plug into the user's ear canal; within that category have been developed cordless air buds using wireless technology. A third type are bone conduction headphones, which typically wrap around the back of the head and rest in front of the ear canal, leaving the ear canal open. In the context of telecommunication, a headset is a combination of a headphone and microphone.
Charger or Chargers may refer to:.
A backpack, also called knapsack, schoolbag, rucksack, pack, booksack, bookbag, haversack, packsack, or backsack, is in its simplest frameless form, a fabric sack carried on one’s back and secured with two straps that go over the shoulders, and is used to carry goods from one place to another. It can feature an external or internal frame to transfer heavy loads off the user’s shoulders and onto their hips, reducing strain and increasing comfort on long hikes with heavy gear.
A wallet is a flat case or pouch, often used to carry small personal items such as physical currency, debit cards, and credit cards; identification documents such as driving licence, identification card, club card; photographs, transit pass, business cards and other paper or laminated cards. Wallets are generally made of fabric or leather, and they are usually pocket-sized and foldable.
Key, Keys, The Key or The Keys may refer to:.
A pen is a common writing instrument that applies ink to a surface, typically paper, for writing or drawing. Early pens such as reed pens, quill pens, dip pens and ruling pens held a small amount of ink on a nib or in a small void or cavity that had to be periodically recharged by dipping the tip of the pen into an inkwell.
Today, such pens find only a small number of specialized uses, such as in illustration and calligraphy. Reed pens, quill pens and dip pens, which were used for writing, have been replaced by ballpoint pens, rollerball pens, fountain pens and felt or ceramic tip pens.
A pencil is a writing or drawing implement with a solid pigment core in a protective casing that reduces the risk of core breakage and keeps it from marking the user's hand.
A notebook is a book or stack of paper pages that are often ruled and used for purposes such as note-taking, journaling, or other writing, drawing, or scrapbooking and more.



Paper is a thin sheet of matted cellulose fibers. Largely derived from lignocellulose, paper is created from a pulp dissolved into a slurry that is drained and dried into sheets. Different types of paper are defined by constituent fiber, paper pulp, sizing, coating, paper size, paper density and grammage.
An eraser is an article of stationery that is used for removing marks from paper or skin. Erasers have a rubbery consistency and come in a variety of shapes, sizes, and colors. Some pencils have an eraser on one end. Less expensive erasers are made from synthetic rubber and synthetic soy-based gum, but more expensive or specialized erasers are made from vinyl, plastic, or gum-like materials.
A highlighter, also called a fluorescent pen, is a type of writing device used to bring attention to sections of text by marking them with a vivid, translucent colour.
A typical highlighter is fluorescent yellow, with the colour coming from pyranine. Different compounds, such as rhodamines are used for other colours.
A ruler is an instrument used to make length measurements, whereby a length is read from a series of markings called "rules" along an edge of the device. Alternatively, it is called a rule, scale, line gauge, or metre/meter stick. Usually, the instrument is rigid and the edge itself is a straightedge, which additionally allows one to draw straighter lines. Rulers are an important tool in geometry, geography and mathematics. They have been used since at least 2650 BC.
Scissors or shears are hand-operated cutting tools that consists of a pair of pivoting blades whose sharpened edges slide firmly against and past each other when the handles (shank) on the opposite side of the pivot are squeezed shut, causing the target material in between the blades to be divided by the combined effort of both cutting and shearing. Scissors are usually used for cutting thin materials such as paper, cardboard, metal foil, cloth, rope and wire, although a large variety of scissors/shears exist for specialized purposes, and their design details often dictate which is best for the intended job.
Tape or Tapes may refer to:.
A stapler is a mechanical device that joins pages of paper or similar material together by driving a thin metal staple through the sheets and folding the ends. Staplers are widely used in government, business, offices, workplaces, homes, and schools.
A mug is a type of cup, a drinking vessel usually intended for hot drinks such as coffee, hot chocolate, or tea. Mugs have handles and usually hold a larger amount of fluid than other types of cups such as teacups or coffee cups. Typically, a mug holds approximately 250–350 ml (8–12 US fl oz) of liquid. A mug-shaped vessel much larger than this tends to be called a tankard.
A cup is a small container used to hold liquids for drinking, typically with a flattened hemispherical shape and an open "mouth", and often with a capacity of about 6–16 US fluid ounces (177–473 ml). Cups may be made of pottery, glass, metal, wood, stone, polystyrene, plastic, lacquerware, or other materials. Normally, a cup is brought in contact with the mouth for drinking, distinguishing it from other tableware and drinkware forms such as jugs; however, a straw and/or lid may also be used. They also often have handles, though many do not, including beakers which have no handle or stem, or small bowl shapes which are very common in Asia.
Plate may refer to:.
A bowl is a typically round dish or container generally used for preparing, serving, storing, or consuming food. The interior of a bowl is characteristically shaped like a spherical cap, with the edges and the bottom, forming a seamless curve. This makes bowls especially suited for holding liquids and loose food, as the contents of the bowl are naturally concentrated in its center by the force of gravity. The exterior of a bowl is typically round but may vary in shape, including rectangular designs.
In cutlery or kitchenware, a fork is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines with which one can spear foods either to hold them to cut with a knife or to lift them to the mouth.
A spoon is a utensil consisting of a shallow bowl, oval or round, at the end of a handle. A type of cutlery, especially as part of a place setting, it is used primarily for transferring food to the mouth (eating). Spoons are also used in food preparation to measure, mix, stir and toss ingredients and for serving food. Present day spoons are made from metal, wood, porcelain or plastic. There are many different types of spoons made from different materials by different cultures for different purposes and food.
A knife is a tool or weapon with a cutting edge or blade, usually attached to a handle or hilt. One of the earliest tools used by humanity, knives appeared at least 2.5 million years ago, as evidenced by the Oldowan tools. Originally made of wood, bone, and stone, over the centuries, in step with improvements in both metallurgy and manufacturing, knife blades have been made from copper, bronze, iron, steel, ceramic, and titanium. Most modern knives have fixed or folding blades, with styles varying by maker and country.
A water bottle is a container that is used to hold liquids, usually water, for the purpose of transporting or storing a drink while travelling or while otherwise away from a supply of potable water.
A vacuum flask is an insulating storage vessel that slows the speed at which its contents change in temperature. It greatly lengthens the time over which its contents remain hotter or cooler than the flask's surroundings by trying to be as adiabatic as possible. Invented by James Dewar in 1892, the vacuum flask consists of two flasks, placed one within the other and joined at the neck. The gap between the two flasks is partially evacuated of air, creating a near-vacuum which significantly reduces heat transfer by conduction or convection. When used to hold cold liquids, this also virtually eliminates condensation on the outside of the flask.
An umbrella is a folding canopy supported by wooden or metal ribs that is mounted on a wooden, metal, or plastic pole. It is usually designed to protect a person against sun or rain. Initially they were used in warmer countries for shade from the sun, but in modern times they evolved to also be used for protection from rain. Etymologically, the term umbrella is to be used when protecting from the sun, but is also commonly used when protecting from rain. Some countries specifically use the words parasol and parapluie to differentiate based on their use. There are also combinations of parasol and parapluie that are called en-tout-cas. A modern hand-held umbrella or parasol may have a black exterior canopy and a silver inner coating, for better protection from both the sun and ultraviolet rays, and may be water-resistant.
A jacket is a garment for the upper body, usually extending below the hips. A jacket typically has sleeves and fastens in the front or slightly on the side. Jackets without sleeves are vests. A jacket is generally lighter, tighter-fitting, and less insulating than a coat, but both are outerwear. Some jackets are fashionable, while some others serve as protective clothing.
A shoe is an item of footwear normally found in pairs intended to protect and comfort the human foot, usually made in such a way that one is designed to fit the left foot and the other the right foot.
A sock is a piece of clothing worn on the feet and often covering the ankle or some part of the calf. Some types of shoes or boots are typically worn over socks. In ancient times, socks were made from leather or matted animal hair. Machine-knit socks were first produced in the late 16th century. Until the 1800s, both hand-made and machine-knit socks were manufactured, with the latter technique becoming more common in the 19th century, and continuing until the modern day.
A hat is a head covering which is worn for various reasons, including protection against weather conditions, ceremonial reasons such as university graduation, religious reasons, comedy, safety, or as a fashion accessory. Hats which incorporate mechanical features, such as visors, spikes, flaps, braces or beer holders shade into the broader category of headgear.
A glove is a garment covering the hand, with separate sheaths or openings for each finger including the thumb. Gloves protect and comfort hands against cold or heat, damage by friction, abrasion or chemicals, and disease; or in turn to provide a guard for what a bare hand should not touch.
Sunglasses or sun glasses are a form of protective eyewear designed primarily to prevent bright sunlight and high-energy visible light from damaging or discomforting the eyes. They can sometimes also function as a visual aid, as variously termed spectacles or glasses exist, featuring lenses that are colored, polarized or darkened. In the early 20th century, they were also known as sun cheaters.
A watch is a timepiece carried or worn by a person. It is designed to maintain a consistent movement despite the motions caused by the person's activities. A wristwatch is worn around the wrist, attached by a watch strap or another type of bracelet, including metal bands or leather straps. A pocket watch is carried in a pocket, often attached to a chain. A stopwatch is a type of watch that measures intervals of time.
A remote control, also known colloquially as a remote or clicker, is an electronic device used to operate another device from a distance, usually wirelessly. In consumer electronics, a remote control can be used to operate devices such as a television set, DVD player or other digital home media appliance. A remote control can allow operation of devices that are out of convenient reach for direct operation of controls. They function best when used from a short distance. This is primarily a convenience feature for the user. In some cases, remote controls allow a person to operate a device that they otherwise would not be able to reach, as when a garage door opener is triggered from outside.
In electrical wiring, a light switch is a switch most commonly used to operate electric lights, permanently connected equipment, or electrical outlets. Portable lamps such as table lamps may have a light switch mounted on the socket, base, or in-line with the cord. Manually operated on/off switches may be substituted by dimmer switches that allow controlling the brightness of lamps as well as turning them on or off, time-controlled switches, occupancy-sensing switches, and remotely controlled switches and dimmers. Light switches are also found in flashlights, vehicles, and other devices.
Lamp, Lamps or LAMP may refer to:.
A pillow is a support of the body at rest for comfort, therapy, or decoration. Pillows are used in different variations by many species, including humans. Some types of pillows include throw pillows, body pillows, decorative pillows, and many more. Pillows that aid sleeping are a form of bedding that supports the head and neck. Other types of pillows are designed to support the body when lying down or sitting. There are also pillows that consider human body shape for increased comfort during sleep. Decorative pillows used on beds, couches or chairs are sometimes referred to as cushions.
A blanket is a swath of soft cloth large enough either to cover or to enfold most of the user's body and thick enough to keep the body warm by trapping radiant body heat that otherwise would be lost through convection and radiation.
Could not find summary for "Bed Sheet".
A towel is a piece of absorbent cloth, or paper, used for drying or wiping a surface. Towels draw moisture through direct contact.
A toothbrush is a special type of brush used to clean the teeth, gums, and tongue. It consists of a head of tightly clustered bristles, onto which toothpaste is applied, mounted on a handle that facilitates cleaning hard-to-reach areas of the mouth. They should be used in conjunction with tools that clean between the teeth―where toothbrush bristles cannot reach―such as floss, tape, interdental brushes or toothpicks.
Toothpaste is a paste or gel dentifrice that is used with a toothbrush to clean and maintain the aesthetics of teeth. Toothpaste is used to promote oral hygiene: it is an abrasive that aids in removing dental plaque and food from the teeth, assists in suppressing halitosis, and delivers active ingredients to help prevent tooth decay and gum disease (gingivitis). Due to variations in composition and fluoride content, not all toothpastes are equally effective in maintaining oral health. The decline of tooth decay during the 20th century has been attributed to the introduction and regular use of fluoride-containing toothpastes worldwide. Large amounts of swallowed toothpaste can be poisonous. Common colors for toothpaste include white and blue.
Soap is a salt of a fatty acid used for cleaning and lubricating products as well as other applications. In a domestic setting, soaps, specifically "toilet soaps", are surfactants usually used for washing, bathing, and other types of housekeeping. In industrial settings, soaps are used as thickeners, components of some lubricants, emulsifiers, and catalysts.
Shampoo is a hair care product, typically in the form of a viscous liquid, that is formulated to be used for cleaning (scalp) hair. Less commonly, it is available in solid bar format. Shampoo is used by applying it to wet hair, massaging the product in the hair, roots and scalp, and then rinsing it out. Some users may follow a shampooing with the use of hair conditioner.
A conditioner is something that improves the quality of another item.
A hairbrush is a brush with rigid or light and soft spokes used in hair care for smoothing, styling, and detangling human hair, or for grooming an animal's fur. It can also be used for styling in combination with a curling iron or hair dryer.
A comb is a tool consisting of a shaft that holds a row of teeth for pulling through the hair to clean, untangle, or style it. Combs have been used since prehistoric times, having been discovered in very refined forms from settlements dating back to 5,000 years ago in Persia.

A deodorant is a substance applied to the body to prevent or mask body odor caused by bacterial breakdown of perspiration, such as that in the armpits, groin, or feet. A subclass of deodorants called antiperspirants prevents sweating itself, typically by blocking sweat glands. Antiperspirants are used on a wider range of body parts at any place where sweat would be inconvenient or unsafe. Other types of deodorant allow sweating but prevent bacterial action on sweat.
A razor is a bladed tool primarily used in the removal of body hair through the act of shaving. Kinds of razors include straight razors, safety razors, disposable razors, and electric shavers.
A mirror, also known as a looking glass, is an object that reflects an image. Light that bounces off a mirror forms an image of whatever is in front of it, which is then focused through the lens of the eye or a camera. Mirrors reverse the direction of light at an angle equal to its incidence. This allows the viewer to see themselves or objects behind them, or even objects that are at an angle from them but out of their field of view, such as around a corner. Natural mirrors have existed since prehistoric times, such as the surface of water, but people have been manufacturing mirrors out of a variety of materials for thousands of years, like stone, metals, and glass. In modern mirrors, metals like silver or aluminium are often used due to their high reflectivity, applied as a thin coating on glass because of its naturally smooth and very hard surface.
A waste container, also known as a dustbin, rubbish bin, trash can, garbage can, wastepaper basket, and wastebasket, among other names, is a type of container intended to store waste. It is usually made out of metal or plastic. The words "rubbish", "basket" and "bin" are more common in British English usage; "trash" and "can" are more common in American English usage. "Garbage" may refer to food waste specifically or to municipal solid waste in general. The word "dumpster" refers to a large outdoor waste container for garbage collectors to pick up the contents.
A recycling bin is a container used to hold recyclables before they are taken to recycling centers. Recycling bins exist in various sizes for use inside and outside of homes, offices, and large public facilities. Separate containers are often provided for paper, tin or aluminum cans, and glass or plastic bottles, with some bins allowing for commingled, mixed recycling of various materials.
A broom, also known as a broomstick, is a cleaning tool, consisting of usually stiff fibers attached to, and roughly parallel to, a cylindrical handle, the broomstick. It is thus a variety of brush with a long handle. It is commonly used in combination with a dustpan.
A dustpan, the small version of which is also known as a "hearth brush and shovel”, is a cleaning utensil. The dustpan is commonly used in combination with a broom or long brush. The small dustpan may appear to be a type of flat scoop. Though often hand-held for home use, industrial and commercial enterprises use a hinged variety on the end of a long handle to allow the user to stand instead of stoop while using it.
A vacuum is space devoid of matter. The word is derived from the Latin adjective vacuus meaning "vacant" or "void". An approximation to such vacuum is a region with a gaseous pressure much less than atmospheric pressure. Physicists often discuss ideal test results that would occur in a perfect vacuum, which they sometimes simply call "vacuum" or free space, and use the term partial vacuum to refer to an actual imperfect vacuum as one might have in a laboratory or in space. In engineering and applied physics on the other hand, vacuum refers to any space in which the pressure is considerably lower than atmospheric pressure. The Latin term in vacuo is used to describe an object that is surrounded by a vacuum.
Could not find summary for "Laundry Basket".
Hanger or hangers may refer to:.
Iron is a chemical element; it has symbol Fe and atomic number 26. It is a metal that belongs to the first transition series and group 8 of the periodic table. It is, by mass, the most common element on Earth, forming much of Earth's outer and inner core. It is the fourth most abundant element in the Earth's crust. In its metallic state it was mainly deposited by meteorites.
Could not find summary for "Ironing Board".
A clock or chronometer is a device that measures and displays time. The clock is one of the oldest human inventions, meeting the need to measure intervals of time shorter than the natural units such as the day, the lunar month, and the year. Devices operating on several physical processes have been used over the millennia.
A calendar is a system of organizing days. This is done by giving names to periods of time, typically days, weeks, months and years. A date is the designation of a single and specific day within such a system. A calendar is also a physical record of such a system. A calendar can also mean a list of planned events, such as a court calendar, or a partly or fully chronological list of documents, such as a calendar of wills.
A whiteboard is a glossy, usually white surface for making non-permanent markings. Whiteboards are analogous to blackboards, but with a smoother surface allowing for rapid marking and erasing of markings on their surface. The popularity of whiteboards increased rapidly in the mid-1990s and they have become a fixture in many offices, meeting rooms, school classrooms, public events and other work environments.
The term Marker may refer to:.
Could not find summary for "Phone Case".
Could not find summary for "Screen Protector".
Could not find summary for "USB Cable".
Could not find summary for "Power Bank".
A flashlight or electric torch, usually shortened to torch, is a portable hand-held electric lamp. Formerly, the light source typically was a miniature incandescent light bulb, but these have been displaced by light-emitting diodes (LEDs) since the early 2000s. A typical flashlight consists of the light source mounted in a reflector, a transparent cover to protect the light source and reflector, a battery, and a switch, all enclosed in a case.
Battery most often refers to:Electric battery, a device that provides electrical power
Battery (crime), a crime involving unlawful physical contact.
Fan commonly refers to:Fan (machine), a machine for producing airflow, often used for cooling
Hand fan, an implement held and waved by hand to move air for cooling
Fan (person), short for fanatic; an enthusiast or supporter, especially with regard to entertainment.
Heating, ventilation, and air conditioning systems use advanced technologies to regulate temperature, humidity, and indoor air quality in residential, commercial, and industrial buildings, and in enclosed vehicles. Its goal is to provide thermal comfort and remove contaminants from the air. HVAC system design is a subdiscipline of mechanical engineering, based on the principles of thermodynamics, fluid mechanics, and heat transfer. Modern HVAC designs focus on energy efficiency and sustainability, especially with the rising demand for green building solutions. In modern construction, MEP engineers integrate HVAC systems with energy modeling techniques to optimize system performance and reduce operational costs. "Refrigeration" is sometimes added to the field's abbreviation as HVAC&R or HVACR, or "ventilation" is dropped, as in HACR.
Air conditioning, often abbreviated as A/C (US) or air con (UK), is the process of removing heat from an enclosed space to achieve a more comfortable interior temperature and, in some cases, controlling the humidity of internal air. Air conditioning can be achieved using a mechanical 'air conditioner' or through other methods, such as passive cooling and ventilative cooling. Air conditioning is a member of a family of systems and techniques that provide heating, ventilation, and air conditioning (HVAC). Heat pumps are similar in many ways to air conditioners but use a reversing valve, allowing them to both heat and cool an enclosed space.
A remote control is any device used to control a remote operation.
Router may refer to:Router (computing), a computer networking device
Router (woodworking), a rotating cutting tool
Router plane, a woodworking hand plane
Journey planner, a specialized search engine for optimal routes between locations
Michael Router, Catholic bishop in Ireland
The Routers, 1960s American instrumental group.
A modulator-demodulator, commonly referred to as a modem, is a computer hardware device that converts data from a digital format into a format suitable for an analog transmission medium such as telephone or radio. A modem transmits data by modulating one or more carrier wave signals to encode digital information, while the receiver demodulates the signal to recreate the original digital information. The goal is to produce a signal that can be transmitted easily and decoded reliably. Modems can be used with almost any means of transmitting analog signals, from LEDs to radio.
Speaker most commonly refers to:Speaker, a person who produces speech
Loudspeaker, a device that produces sound
Computer speakers.
A camera is an instrument used to capture and store images and videos, either digitally via an electronic image sensor, or chemically via a light-sensitive material such as photographic film. As a pivotal technology in the fields of photography and videography, cameras have played a significant role in the progression of visual arts, media, entertainment, surveillance, and scientific research. The invention of the camera dates back to the 19th century and has since evolved with advancements in technology, leading to a vast array of types and models in the 21st century.
A tripod is a portable three-legged frame or stand, used as a platform for supporting the weight and maintaining the stability of some other object. The three-legged design provides good stability against gravitational loads as well as horizontal shear forces, and better leverage for resisting tipping over due to lateral forces can be achieved by spreading the legs away from the vertical centre.
Variations with one, two, and four legs are termed monopod, bipod, and quadripod.
A microphone, colloquially called a mic, or mike, is a transducer that converts sound into an electrical signal. Microphones are used in telecommunication, sound recording, broadcasting, and consumer electronics, including telephones, hearing aids, and mobile devices.
Notebook Paper is the debut studio album by American rapper Huey. It was released on June 19, 2007, via Hitz Committee/Jive/Zomba Records. Production was handled by several record producers, including Jazze Pha, StarGate, T-Mix and T-Pain. It features guest appearances from Asia Cruise, Diamond, Kydd Trell, Bow Wow, Lloyd, MeMpHiTz, T-Pain, Trey Songz and Yo Gotti.
Sticky Notes is a desktop notes application included in Windows 7, Windows 8, Windows 8.1, Windows 10 and Windows 11. The app loads quickly and enables users to quickly take notes using post-it note–like windows on their desktop.
An envelope is a common packaging item, usually made of thin, flat material. It is designed to contain a flat object, such as a letter or card.
Stamp or Stamps or Stamping may refer to:.
Could not find summary for "Wallet Card".
An identity document is a document proving a person's identity.
A coin is a small object, usually round and flat, used primarily as a medium of exchange or legal tender. They are standardized in weight, and produced in large quantities at a mint in order to facilitate trade. They are most often issued by a government. Coins often have images, numerals, or text on them. The faces of coins or medals are sometimes called the obverse and the reverse, referring to the front and back sides, respectively. The obverse of a coin is commonly called heads, because it often depicts the head of a prominent person, and the reverse is known as tails.
Could not find summary for "Water Filter".
Could not find summary for "Dish Sponge".
Could not find summary for "Cutting Board".
Pan or PAN may refer to:.
Pot may refer to:.
Could not find summary for "Oven Mitt".
A measuring cup is a kitchen utensil used primarily to measure the volume of liquid or bulk solid cooking ingredients such as flour and sugar, especially for volumes from about 50 mL upwards. Measuring cups are also used to measure washing powder, liquid detergents and bleach for clothes washing. Some measuring cups will have a scale marked in cups and fractions of a cup, and often with fluid measure and weight of a selection of dry foodstuffs. Others are made to a specific capacity and are designed to be filled to the top with dry ingredients.
Could not find summary for "Measuring Spoon".
Quantum mechanics is the fundamental physical theory that describes the behavior of matter and of light; its unusual characteristics typically occur at and below the scale of atoms. It is the foundation of all quantum physics, which includes quantum chemistry, quantum biology, quantum field theory, quantum technology, and quantum information science.
General relativity, also known as the general theory of relativity, and as Einstein's theory of gravity, is the geometric theory of gravitation published by Albert Einstein in May 1916 and is the accepted description of the gravitation of macroscopic objects in modern physics. General relativity generalizes special relativity and refines Isaac Newton's law of universal gravitation, providing a unified description of gravity as a geometric property of space and time, or four-dimensional spacetime. In particular, the curvature of spacetime is directly related to the energy, momentum, and stress of whatever is present, including matter and radiation. The relation is specified by the Einstein field equations, a system of second-order partial differential equations. John Archibald Wheeler summarized it: "Space-time tells matter how to move; matter tells space-time how to curve.".
In physics, the special theory of relativity, or special relativity for short, is a scientific theory of the relationship between space and time. In Albert Einstein's 1905 paper,
"On the Electrodynamics of Moving Bodies", the theory is presented as being based on just two postulates:The laws of physics are invariant (identical) in all inertial frames of reference. This is known as the principle of relativity.
The speed of light in vacuum is the same for all observers, regardless of the motion of light source or observer. This is known as the principle of light constancy, or the principle of light speed invariance.
In physics, classical mechanics is a theory that describes the effect of forces on the motion of macroscopic objects and bulk matter, without considering quantum effects, and often without incorporating relativistic effects either.
Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation. The behavior of these quantities is governed by the four laws of thermodynamics, which convey a quantitative description using measurable macroscopic physical quantities but may be explained in terms of microscopic constituents by statistical mechanics. Thermodynamics applies to various topics in science and engineering, especially physical chemistry, biochemistry, chemical engineering, and mechanical engineering, as well as other complex fields such as meteorology.
In physics, statistical mechanics is a mathematical framework that applies statistical methods and probability theory to large assemblies of microscopic entities. Sometimes called statistical physics or statistical thermodynamics, its applications include many problems in a wide variety of fields such as biology, neuroscience, computer science, information theory and sociology. Its main purpose is to clarify the properties of matter in aggregate, in terms of physical laws governing atomic motion.
In physics, electromagnetism is an interaction that occurs between particles with electric charge via electromagnetic fields. The electromagnetic force is one of the four fundamental forces of nature. It is the dominant force in the interactions of atoms and molecules. Electromagnetism can be thought of as a combination of electrostatics and magnetism, which are distinct but closely intertwined phenomena. Electromagnetic forces occur between any two charged particles. Electric forces cause an attraction between particles with opposite charges and repulsion between particles with the same charge, while magnetism is an interaction that occurs between charged particles in relative motion. These two forces are described in terms of electromagnetic fields. Macroscopic charged objects are described in terms of Coulomb's law for electricity and Ampère's force law for magnetism; the Lorentz force describes microscopic charged particles.
In theoretical physics, quantum field theory (QFT) is a theoretical framework that combines field theory, special relativity and quantum mechanics. QFT is used in particle physics to construct physical models of subatomic particles and in condensed matter physics to construct models of quasiparticles. The current standard model of particle physics is based on QFT.
Particle physics or high-energy physics is the study of fundamental particles and forces that constitute matter and radiation. The field also studies combinations of elementary particles up to the scale of protons and neutrons, while the study of combinations of protons and neutrons is called nuclear physics.
Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions, in addition to the study of other forms of nuclear matter.
Astrophysics is a science that applies the methods and principles of physics and chemistry in the study of astronomical objects and phenomena including the universe. As one of the founders of the discipline, James Keeler, said, astrophysics "seeks to ascertain the nature of the heavenly bodies, rather than their positions or motions in space—what they are, rather than where they are", which is studied in celestial mechanics.
Cosmology is the study of the nature of the universe, the cosmos. The term cosmology was first used in English in 1656 in Thomas Blount's Glossographia, with the meaning of "a speaking of the world". In 1731, German philosopher Christian Wolff used the term cosmology in Latin (cosmologia) to denote a branch of metaphysics that deals with the general nature of the physical world. Cosmology is investigated by scientists, including astronomers and physicists, as well as philosophers, such as metaphysicians, philosophers of physics, and philosophers of space and time. Because of this shared scope with philosophy, theories in physical cosmology may include both scientific and non-scientific propositions and may depend upon assumptions that cannot be tested. Religious or mythological cosmology is a body of beliefs based on mythological, religious, and esoteric literature and traditions of creation myths and eschatology.
Stellar evolution is the process by which a star changes over the course of time. Depending on the mass of the star, its lifetime can range from a few million years for the most massive to trillions of years for the least massive, which is considerably longer than the current age of the universe. The table shows the lifetimes of stars as a function of their masses. All stars are formed from collapsing clouds of gas and dust, often called nebulae or molecular clouds. Over the course of millions of years, these protostars settle down into a state of equilibrium, becoming what is known as a main sequence star.
Planetary science is the scientific study of planets, celestial bodies and planetary systems and the processes of their formation. It studies objects ranging in sizes from micrometeoroids to huge gas giants, with the aim of determining their composition, dynamics, formation, interrelations and history. It is a strongly interdisciplinary field, which originally grew from astronomy and Earth science, and now incorporates many disciplines, including planetary geology, cosmochemistry, atmospheric science, physics, oceanography, hydrology, theoretical planetary science, glaciology, and exoplanetology. Allied disciplines include space physics, when concerned with the effects of the Sun on the bodies of the Solar System, and astrobiology.
Could not find summary for "Exoplanet Research".
Could not find summary for "Galactic Dynamics".
Could not find summary for "Black Hole Physics".
In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings. String theory describes how these strings propagate through space and interact with each other. On distance scales larger than the string scale, a string acts like a particle, with its mass, charge, and other properties determined by the vibrational state of the string. In string theory, one of the many vibrational states of the string corresponds to the graviton, a quantum mechanical particle that carries the gravitational force. Thus, string theory is a theory of quantum gravity.
Chaos theory is an interdisciplinary area of scientific study and branch of mathematics. It focuses on underlying patterns and deterministic laws of dynamical systems that are highly sensitive to initial conditions. These were once thought to have completely random states of disorder and irregularities. Chaos theory states that within the apparent randomness of chaotic complex systems, there are underlying patterns, interconnection, constant feedback loops, repetition, self-similarity, fractals and self-organization. The butterfly effect, an underlying principle of chaos, describes how a small change in one state of a deterministic nonlinear system can result in large differences in a later state. A metaphor for this behavior is that a butterfly flapping its wings in Brazil can cause or prevent a tornado in Texas.
A complex system is a system composed of many components that interact with one another. Examples of complex systems are Earth's global climate, organisms, the human brain, infrastructure such as power grid, transportation or communication systems, complex software and electronic systems, social and economic organizations, an ecosystem, a living cell, and, ultimately, for some authors, the entire universe.
Evolutionary biology is a subfield of biology that analyzes the four mechanisms of evolution: natural selection, mutation, genetic drift, and gene flow. The purpose of evolutionary biology is to observe the diversity of life on Earth. The idea of natural selection was first researched by Charles Darwin as he studied bird beaks. The discipline of evolutionary biology emerged through what Julian Huxley called the modern synthesis of understanding, from previously unrelated fields of biological research, such as genetics and ecology, systematics, and paleontology. Huxley was able to take what Charles Darwin discovered and elaborate to build on his understandings.
Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene.
Molecular biology is a branch of biology that seeks to understand the molecular structures and chemical processes that are the basis of biological activity within and between cells. It is centered largely on the study of nucleic acids and proteins. It examines the structure, function, and interactions of these macromolecules as they orchestrate processes such as replication, transcription, translation, protein synthesis, and complex biomolecular interactions. The field of molecular biology is multi-disciplinary, relying on principles from genetics, biochemistry, physics, mathematics, and more recently computer science (bioinformatics).
Cell biology, cellular biology, or cytology, is the branch of biology that studies the structure, function, and behavior of the cells. All organisms are made of cells. A cell is the basic unit of life that is responsible for the living and functioning of an organism. Cell biology encompasses both prokaryotic and eukaryotic cells, with subtopics including the study of cell metabolism, cell communication, cell cycle, biochemistry, and cell composition.
Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences.

Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.".
Biochemistry, or biological chemistry, is the study of chemical processes within and relating to living organisms. A sub-discipline of both chemistry and biology, biochemistry may be divided into three fields: structural biology, enzymology, and metabolism. Over the last decades of the 20th century, biochemistry has become successful at explaining living processes through these three disciplines. Almost all areas of the life sciences are being uncovered and developed through biochemical methodology and research. Biochemistry focuses on understanding the chemical basis that allows biological molecules to give rise to the processes that occur within living cells and between cells, in turn relating greatly to the understanding of tissues and organs as well as organism structure and function. Biochemistry is closely related to molecular biology, the study of the molecular mechanisms of biological phenomena.
Biophysics is an interdisciplinary science that applies approaches and methods traditionally used in physics to study biological phenomena.
Microbiology is the scientific study of microorganisms, those being of unicellular (single-celled), multicellular, or acellular. Microbiology encompasses numerous sub-disciplines including virology, bacteriology, protistology, mycology, immunology, and parasitology.
Virology is the scientific study of biological viruses. It is a subfield of microbiology that focuses on their detection, structure, classification and evolution, their methods of infection and exploitation of host cells for reproduction, their interaction with host organism physiology and immunity, the diseases they cause, the techniques to isolate and culture them, and their use in research and therapy.
Immunology is a branch of biology and medicine that covers the study of immune systems in all organisms.
Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history.
Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology.
Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry.
Oceanography, also known as oceanology, sea science, ocean science, and marine science, is the scientific study of the ocean, including its physics, chemistry, biology, and geology.
Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science.
Volcanology is the study of volcanoes, lava, magma and related geological, geophysical and geochemical phenomena (volcanism). The term volcanology is derived from the Latin word vulcan. Vulcan was the ancient Roman god of fire.
Seismology is the scientific study of earthquakes and the generation and propagation of elastic waves through planetary bodies. It also includes studies of the environmental effects of earthquakes such as tsunamis; other seismic sources such as volcanoes, plate tectonics, glaciers, rivers, oceanic microseisms, and the atmosphere; and artificial processes such as explosions.
Paleontology or palaeontology is the scientific study of the life of the past, mainly but not exclusively through the study of fossils. Paleontologists use fossils as a means to classify organisms, measure geologic time, and assess the interactions between prehistoric organisms and their natural environment. While paleontological observations are known from at least the 6th century BC, the foundation of paleontology as a science dates back to the work of Georges Cuvier in 1796. Cuvier demonstrated evidence for the concept of extinction and how the life of the past was not necessarily the same as that of the present. The field developed rapidly over the course of the following decades, and the French word paléontologie was introduced for the study in 1822, which was derived from the Ancient Greek word for 'ancient' and words describing relatedness and a field of study. Further advances in the field accompanied the work of Charles Darwin who popularized the concept of evolution. Together, evolution and extinction can be understood as complementary processes that shaped the history of life.
Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research.
Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives.
Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
Polymer science or macromolecular science is a subfield of materials science concerned with polymers, primarily synthetic polymers such as plastics and elastomers. The field of polymer science includes researchers in multiple disciplines including chemistry, physics, and engineering.
Crystallography is the branch of science devoted to the study of molecular and crystalline structure and properties. The word crystallography is derived from the Ancient Greek word κρύσταλλος, and γράφειν. In July 2012, the United Nations recognised the importance of the science of crystallography by proclaiming 2014 the International Year of Crystallography.
Organic chemistry is a subdiscipline within chemistry involving the scientific study of the structure, properties, and reactions of organic compounds and organic materials. It involves studying the structure of organic material to determine the structural formula, analyzing physical and chemical properties, and evaluating chemical reactivity to understand the behavior of organic compounds. The study of organic reactions includes the chemical synthesis of natural products, drugs, and polymers, and study of individual organic molecules in the laboratory and via theoretical study.
Inorganic chemistry deals with synthesis and behavior of inorganic and organometallic compounds. This field covers chemical compounds that are not carbon-based, which are the subjects of organic chemistry. The distinction between the two disciplines is far from absolute, as there is much overlap in the subdiscipline of organometallic chemistry. It has applications in every aspect of the chemical industry, including catalysis, materials science, pigments, surfactants, coatings, medications, fuels, and agriculture.
Physical chemistry is the study of macroscopic and microscopic phenomena in chemical systems in terms of the principles, practices, and concepts of physics such as motion, energy, force, time, thermodynamics, quantum chemistry, statistical mechanics, analytical dynamics and chemical equilibria.
Analytical chemistry is the branch of chemistry concerned with the development and application of methods to identify the chemical composition of materials and quantify the amounts of components in mixtures. It focuses on methods to identify unknown compounds, possibly in a mixture or solution, and quantify a compound's presence in terms of amount of substance, concentration, percentage by mass or number of moles in a mixture of compounds.
Computational chemistry is a branch of chemistry that uses computer simulations to assist in solving chemical problems. It uses methods of theoretical chemistry incorporated into computer programs to calculate the structures and properties of molecules, groups of molecules, and solids. The importance of this subject stems from the fact that, with the exception of some relatively recent findings related to the hydrogen molecular ion, achieving an accurate quantum mechanical depiction of chemical systems analytically, or in a closed form, is not feasible. The complexity inherent in the many-body problem exacerbates the challenge of providing detailed descriptions of quantum mechanical systems. While computational results normally complement information obtained by chemical experiments, it can occasionally predict unobserved chemical phenomena.
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
In machine learning, deep learning (DL) focuses on utilizing multilayered neural networks to perform tasks such as classification, regression, and representation learning. The field takes inspiration from biological neuroscience and revolves around stacking artificial neurons into layers and "training" them to process data. The adjective "deep" refers to the use of multiple layers in the network. Methods used can be supervised, semi-supervised or unsupervised.
Computer vision tasks include methods for acquiring, processing, analyzing, and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, e.g. in the form of decisions. "Understanding" in this context signifies the transformation of visual images into descriptions of the world that make sense to thought processes and can elicit appropriate action. This image understanding can be seen as the disentangling of symbolic information from image data using models constructed with the aid of geometry, physics, statistics, and learning theory.
Natural language processing (NLP) is the processing of natural language information by a computer. NLP is a subfield of computer science and is closely associated with artificial intelligence. NLP is also related to information retrieval, knowledge representation, computational linguistics, and linguistics more broadly.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. Robotics usually combines four aspects of design work: a power source, mechanical construction, a control system, and software.
Cybernetics is the transdisciplinary study of circular causal processes such as feedback and recursion, where the effects of a system's actions return as inputs to that system, influencing subsequent actions. It is concerned with general principles that are relevant across multiple contexts, including engineering, ecological, economic, biological, cognitive and social systems and also in practical activities such as designing, learning, and managing. Cybernetics' transdisciplinary character means that it intersects with a number of other fields, resulting in a wide influence and diverse interpretations.
Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering.
Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications.
A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems. These systems evolve in ways that operate on an enormous number of possibilities simultaneously, though they remain subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks.
Bioinformatics is an interdisciplinary field of science that develops computational methods and software tools for understanding biological data, especially when the data sets are large and complex. Bioinformatics uses biology, chemistry, physics, computer science, data science, computer programming, information engineering, mathematics and statistics to analyze and interpret biological data. This process can sometimes be referred to as computational biology, however the distinction between the two terms is often disputed. To some, the term computational biology refers to building and using models of biological systems.
Systems biology is the computational and mathematical analysis and modeling of complex biological systems. It is a biology-based interdisciplinary field of study that focuses on complex interactions within biological systems, using a holistic approach to biological research. This multifaceted research domain necessitates the collaborative efforts of chemists, biologists, mathematicians, physicists, and engineers to decipher the biology of intricate living systems by merging various quantitative molecular measurements with carefully constructed mathematical models. It represents a comprehensive method for comprehending the complex relationships within biological systems. In contrast to conventional biological studies that typically center on isolated elements, systems biology seeks to combine different biological data to create models that illustrate and elucidate the dynamic interactions within a system. This methodology is essential for understanding the complex networks of genes, proteins, and metabolites that influence cellular activities and the traits of organisms. One of the aims of systems biology is to model and discover emergent properties, of cells, tissues and organisms functioning as a system whose theoretical description is only possible using techniques of systems biology. By exploring how function emerges from dynamic interactions, systems biology bridges the gaps that exist between molecules and physiological processes.
Synthetic biology (SynBio) is a multidisciplinary field of science that focuses on living systems and organisms. It applies engineering principles to develop new biological parts, devices, and systems or to redesign existing systems found in nature.
Genetic engineering, also called genetic modification or genetic manipulation, is the modification and manipulation of an organism's genes using technology. It is a set of technologies used to change the genetic makeup of cells, including the transfer of genes within and across species boundaries to produce improved or novel organisms. New DNA is obtained by either isolating and copying the genetic material of interest using recombinant DNA methods or by artificially synthesising the DNA. A construct is usually created and used to insert this DNA into the host organism. The first recombinant DNA molecule was designed by Paul Berg in 1972 by combining DNA from the monkey virus SV40 with the lambda virus. As well as inserting genes, the process can be used to remove, or "knock out", genes. The new DNA can either be inserted randomly or targeted to a specific part of the genome.
Could not find summary for "CRISPR Technology".
Pharmacology is the science of drugs and medications, including a substance's origin, composition, pharmacokinetics, pharmacodynamics, therapeutic use, and toxicology. More specifically, it is the study of the interactions that occur between a living organism and chemicals that affect normal or abnormal biochemical function. If substances have medicinal properties, they are considered pharmaceuticals.
Toxicology is a scientific discipline, overlapping with biology, chemistry, pharmacology, and medicine, that involves the study of the adverse effects of chemical substances on living organisms and the practice of diagnosing and treating exposures to toxins and toxicants. The relationship between dose and its effects on the exposed organism is of high significance in toxicology. Factors that influence chemical toxicity include the dosage, duration of exposure, route of exposure, species, age, sex, and environment. Toxicologists are experts on poisons and poisoning. There is a movement for evidence-based toxicology as part of the larger movement towards evidence-based practices. Toxicology is currently contributing to the field of cancer research, since some toxins can be used as drugs for killing tumor cells. One prime example of this is ribosome-inactivating proteins, tested in the treatment of leukemia.
Neuropharmacology is the study of how drugs affect function in the nervous system, and the neural mechanisms through which they influence behavior. There are two main branches of neuropharmacology: behavioral and molecular. Behavioral neuropharmacology focuses on the study of how drugs affect human behavior (neuropsychopharmacology), including the study of how drug dependence and addiction affect the human brain. Molecular neuropharmacology involves the study of neurons and their neurochemical interactions, with the overall goal of developing drugs that have beneficial effects on neurological function. Both of these fields are closely connected, since both are concerned with the interactions of neurotransmitters, neuropeptides, neurohormones, neuromodulators, enzymes, second messengers, co-transporters, ion channels, and receptor proteins in the central and peripheral nervous systems. Studying these interactions, researchers are developing drugs to treat many different neurological disorders, including pain, neurodegenerative diseases such as Parkinson's disease and Alzheimer's disease, psychological disorders, addiction, and many others.
Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole.
Radio astronomy is a subfield of astronomy that studies celestial objects using radio waves. It started in 1933, when Karl Jansky at Bell Telephone Laboratories reported radiation coming from the Milky Way. Subsequent observations have identified a number of different sources of radio emission. These include stars and galaxies, as well as entirely new classes of objects, such as radio galaxies, quasars, pulsars, and masers. The discovery of the cosmic microwave background radiation, regarded as evidence for the Big Bang theory, was made through radio astronomy.
Optics is the branch of physics that studies the behaviour, manipulation, and detection of electromagnetic radiation, including its interactions with matter and instruments that use or detect it. Optics usually describes the behaviour of visible, ultraviolet, and infrared light. The study of optics extends to other forms of electromagnetic radiation, including radio waves, microwaves,
and X-rays. The term optics is also applied to technology for manipulating beams of elementary charged particles.
Photonics is a branch of optics that involves the application of generation, detection, and manipulation of light in the form of photons through emission, transmission, modulation, signal processing, switching, amplification, and sensing. Even though photonics is a commonly used term, there is no widespread agreement on a clear definition of the term or on the difference between photonics and related fields, such as optics.
Acoustics is a branch of continuum mechanics that deals with the study of mechanical waves in gases, liquids, and solids including topics such as vibration, sound, ultrasound and infrasound. A scientist who works in the field of acoustics is an acoustician while someone working in the field of acoustics technology may be called an acoustical engineer. The application of acoustics is present in almost all aspects of modern society with the most obvious being the audio and noise control industries.
In physics, physical chemistry, and engineering, fluid dynamics is a subdiscipline of fluid mechanics that describes the flow of fluids – liquids and gases. It has several subdisciplines, including aerodynamics and hydrodynamics. Fluid dynamics has a wide range of applications, including calculating forces and moments on aircraft, determining the mass flow rate of petroleum through pipelines, predicting weather patterns, understanding nebulae in interstellar space, understanding large scale geophysical flows involving oceans/atmosphere and modelling fission weapon detonation.
Aerodynamics is the study of the motion of air, particularly when affected by a solid object, such as an airplane wing. It involves topics covered in the field of fluid dynamics and its subfield of gas dynamics, and is an important domain of study in aeronautics. The term aerodynamics is often used synonymously with gas dynamics, the difference being that "gas dynamics" applies to the study of the motion of all gases, and is not limited to air. The formal study of aerodynamics began in the modern sense in the eighteenth century, although observations of fundamental concepts such as aerodynamic drag were recorded much earlier. Most of the early efforts in aerodynamics were directed toward achieving heavier-than-air flight, which was first demonstrated by Otto Lilienthal in 1891. Since then, the use of aerodynamics through mathematical analysis, empirical approximations, wind tunnel experimentation, and computer simulations has formed a rational basis for the development of heavier-than-air flight and a number of other technologies. Recent work in aerodynamics has focused on issues related to compressible flow, turbulence, and boundary layers and has become increasingly computational in nature.
Plasma is a state of matter that results from a gaseous state having undergone some degree of ionization. It thus consists of a significant portion of charged particles. While rarely encountered on Earth, it is estimated that 99.9% of all ordinary matter in the universe is plasma. Stars are almost pure balls of plasma, and plasma dominates the rarefied intracluster medium and intergalactic medium. Plasma can be artificially generated, for example, by heating a neutral gas or subjecting it to a strong electromagnetic field.
Could not find summary for "Energy Science".
Renewable energy is energy made from renewable natural resources that are replenished on a human timescale. The most widely used renewable energy types are solar energy, wind power, and hydropower. Bioenergy and geothermal power are also significant in some countries. Renewable energy installations can be large or small and are suited for both urban and rural areas. Renewable energy is often deployed together with further electrification. This has several benefits: electricity can move heat and vehicles efficiently and is clean at the point of consumption. Variable renewable energy sources are those that have a fluctuating nature, such as wind power and solar power. In contrast, controllable renewable energy sources include dammed hydroelectricity, bioenergy, or geothermal power.
Nuclear fusion is a reaction in which two or more atomic nuclei combine to form a larger nucleus. The difference in mass between the reactants and products is manifested as either the release or the absorption of energy. This difference in mass arises as a result of the difference in nuclear binding energy between the atomic nuclei before and after the fusion reaction. Nuclear fusion is the process that powers all active stars, via many reaction pathways.
Could not find summary for "Space Engineering".
Aerospace engineering is the primary field of engineering concerned with the development of aircraft and spacecraft. It has two major and overlapping branches: aeronautical engineering and astronautical engineering. Avionics engineering is similar, but deals with the electronics side of aerospace engineering.
The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global.
Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use.
Chemical engineering is an engineering field which deals with the study of the operation and design of chemical plants as well as methods of improving production. Chemical engineers develop economical commercial processes to convert raw materials into useful products. Chemical engineering uses principles of chemistry, physics, mathematics, biology, and economics to efficiently use, produce, design, transport and transform energy and materials. The work of chemical engineers can range from the utilization of nanotechnology and nanomaterials in the laboratory to large-scale industrial processes that convert chemicals, raw materials, living cells, microorganisms, and energy into useful forms and products. Chemical engineers are involved in many aspects of plant design and operation, including safety and hazard assessments, process design and analysis, modeling, control engineering, chemical reaction engineering, nuclear engineering, biological engineering, construction specification, and operating instructions.
Biomedical engineering (BME) or medical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare applications. BME also integrates the logical sciences to advance health care treatment, including diagnosis, monitoring, and therapy. Also included under the scope of a biomedical engineer is the management of current medical equipment in hospitals while adhering to relevant industry standards. This involves procurement, routine testing, preventive maintenance, and making equipment recommendations, a role also known as a Biomedical Equipment Technician (BMET) or as a clinical engineer.
Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways.
Structural engineering is a sub-discipline of civil engineering in which structural engineers are trained to design the 'bones and joints' that create the form and shape of human-made structures. Structural engineers also must understand and calculate the stability, strength, rigidity and earthquake-susceptibility of built structures for buildings and nonbuilding structures. The structural designs are integrated with those of other designers such as architects and building services engineer and often supervise the construction of projects by contractors on site. They can also be involved in the design of machinery, medical equipment, and vehicles where structural integrity affects functioning and safety. See glossary of structural engineering.

A mathematical model is an abstract description of a concrete system using mathematical concepts and language. The process of developing a mathematical model is termed mathematical modeling. Mathematical models are used in many fields, including applied mathematics, natural sciences, social sciences and engineering. In particular, the field of operations research studies the use of mathematical modelling and related tools to solve problems in business or military operations. A model may help to characterize a system by studying the effects of different components, which may be used to make predictions about behavior or solve specific problems.
Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
Number theory is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. Number theorists study prime numbers as well as the properties of mathematical objects constructed from integers, or defined as generalizations of the integers.
Probability theory or probability calculus is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms. Typically these axioms formalise probability in terms of a probability space, which assigns a measure taking values between 0 and 1, termed the probability measure, to a set of outcomes called the sample space. Any specified subset of the sample space is called an event.
Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers.
Econometrics is an application of statistical methods to economic data in order to give empirical content to economic relationships. More precisely, it is "the quantitative analysis of actual economic phenomena based on the concurrent development of theory and observation, related by appropriate methods of inference." An introductory economics textbook describes econometrics as allowing economists "to sift through mountains of data to extract simple relationships." Jan Tinbergen is one of the two founding fathers of econometrics. The other, Ragnar Frisch, also coined the term in the sense in which it is used today.
Social physics or sociophysics is an interdisciplinary field of science which uses mathematical tools inspired by physics to understand the behavior of human crowds. In a modern commercial use, it can also refer to the analysis of social phenomena with big data.

Behavioural science is the branch of science concerned with theorizing on, categorizing, and judging human behaviour. It sits in the interstice between fields such as psychology, cognitive science, neuroscience, behavioral biology, behavioral genetics and social science. While the term can technically be applied to the study of behaviour amongst all living organisms, it is nearly always used with reference to humans as the primary target of investigation.
Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions.
Could not find summary for "Cognitive Robotics".
Astrobiology is a scientific field within the life and environmental sciences that studies the origins, early evolution, distribution, and future of life in the universe by investigating its deterministic conditions and contingent events. As a discipline, astrobiology is founded on the premise that life may exist beyond Earth.
Could not find summary for "Exochemistry".
Egypt, officially the Arab Republic of Egypt, is a country spanning the northeast corner of Africa and southwest corner of Asia via the Sinai Peninsula. It is bordered by the Mediterranean Sea to the north, Palestine and Israel to the northeast, the Red Sea to the east, Sudan and the Sahara to the south, and Libya to the west. The Gulf of Aqaba in the northeast separates Egypt from Jordan and Saudi Arabia. Cairo is the capital, largest city, and leading cultural centre, while Alexandria is the second-largest city and an important hub of industry and tourism. With over 107 million inhabitants, Egypt is the most populous country in the Arab world, third-most populous country in Africa, and 15th-most populated in the world.
Mesopotamia is a historical region of West Asia situated within the Tigris–Euphrates river system, in the northern part of the Fertile Crescent. It corresponds roughly to the territory of modern Iraq. Just beyond it lies southwestern Iran, where the region transitions into the Persian plateau, marking the shift from the Arab world to Iran.
Iran, officially the Islamic Republic of Iran, and also known as Persia, is a country in West Asia. It borders Iraq to the west, Turkey, Azerbaijan, and Armenia to the northwest, the Caspian Sea to the north, Turkmenistan to the northeast, Afghanistan to the east, Pakistan to the southeast, and the Gulf of Oman and the Persian Gulf to the south. With a population of over 90 million, Iran ranks 17th globally in both geographic size and population and is the sixth-largest country in Asia. It is divided into five regions with 31 provinces. Tehran is the nation's capital, largest city, and financial center.
Greece, officially the Hellenic Republic, is a country in Southeast Europe. Located on the southern tip of the Balkan peninsula, it shares land borders with Albania to the northwest, North Macedonia and Bulgaria to the north, and Turkey to the east. The Aegean Sea lies to the east of the mainland, the Ionian Sea to the west, and the Sea of Crete and the Mediterranean Sea to the south. Greece has the longest coastline on the Mediterranean basin, spanning thousands of islands and nine traditional geographic regions. It has a population of over 10 million. Athens is the nation's capital and largest city, followed by Thessaloniki and Patras.
Rome is the capital city and most populated comune (municipality) of Italy. It is also the administrative centre of the Lazio region and of the Metropolitan City of Rome. A special comune named Roma Capitale with a population of 2.7 million in an area of 1,287.36 km2 (497.1 mi2), Rome is the third most populous city in the European Union by population within city limits. The Metropolitan City of Rome Capital, with a population of 4.2 million, is the most populous metropolitan city in Italy. Its metropolitan area is the third-most populous within Italy. Rome is located in the central-western portion of the Italian Peninsula, within Lazio (Latium), along the shores of the Tiber Valley. Vatican City is an independent country inside the city boundaries of Rome, the only existing example of a country within a city. Rome is often referred to as the "City of Seven Hills" due to its geography, and also as the "Eternal City". Rome is generally considered to be one of the cradles of Western civilization and Western Christian culture, and the centre of the Catholic Church.
Byzantium or Byzantion was an ancient Greek city in classical antiquity that became known as Constantinople in late antiquity and Istanbul in modern times. The Greek name Byzantion and its Latinization Byzantium continued to be used as a name of Constantinople sporadically and to varying degrees during the thousand-year existence of the Eastern Roman Empire, which also became known by the former name of the city as the Byzantine Empire. Byzantium was colonized by Greeks from Megara in the 7th century BCE and remained primarily Greek-speaking until its conquest by the Ottoman Empire in 1453 CE.
Ottoman may refer to:Osman I, historically known in English as "Ottoman I", founder of the Ottoman Empire
Osman II, historically known in English as "Ottoman II"
Osman III, historically known in English as "Ottoman III"
Ottoman Empire 1299–1922
Ottoman dynasty, ruling family of the Ottoman Empire
Osmanoğlu family, modern members of the family
Ottoman Caliphate 1517–1924
Ottoman Turks, a Turkic ethnic group
Ottoman architecture
Ottoman bed, a type of storage bed
Ottoman (furniture), padded stool or footstool
Ottoman (textile), fabric with a pronounced ribbed or corded effect, often made of silk or a mixture.
Mongols are an East Asian ethnic group native to Mongolia and China, as well as the republics of Buryatia and Kalmykia in Russia. The Mongols are the principal member of the large family of Mongolic peoples. The Oirats and the Buryats are classified either as distinct ethno-linguistic groups or as subgroups of Mongols.
China, officially the People's Republic of China (PRC), is a country in East Asia. It is the second-most populous country after India, with a population exceeding 1.4 billion, representing 17% of the world's population. China borders fourteen countries by land across an area of 9.6 million square kilometers (3,700,000 sq mi), making it the third-largest country by area. The country is divided into 33 province-level divisions: 22 provinces, 5 autonomous regions, 4 municipalities, and 2 semi-autonomous special administrative regions. Beijing is the capital, while Shanghai is the most populous city by urban area and largest financial center.
Japan is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered to the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. The Japanese archipelago consists of four major islands alongside 14,121 smaller islands. Japan is divided into 47 administrative prefectures and eight traditional regions, and around 75% of its terrain is mountainous and heavily forested, concentrating its agriculture and highly urbanized population along its eastern coastal plains. With a population of almost 123 million as of 2026, it is the world's 11th most populous country. Tokyo is the country's capital and largest city.
Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands. Since the end of World War II in Asia in 1945, it has been politically divided at or near the 38th parallel between North Korea and South Korea. Both countries proclaimed independence in 1948, and the two countries fought the Korean War from 1950 to 1953. The region is bordered by China to the north and Russia to the northeast, across the Amnok (Yalu) and Duman (Tumen) rivers, and is separated from Japan to the southeast by the Korea Strait.
India, officially the Republic of India, is a country in South Asia. It is the seventh-largest country by area; the most populous country since 2023; and, since its independence in 1947, the world's most populous democracy. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand, and Indonesia.
Maya may refer to:.
The Aztecs were a Mesoamerican civilization that flourished in central Mexico from 1300 to 1521. The Aztec people included different ethnic groups of central Mexico, particularly those groups who spoke the Nahuatl language. Aztec culture was organized into city-states (altepetl), some of which joined to form alliances, political confederations, or empires. The Aztec Empire was a confederation of three city-states established in 1427: Tenochtitlan, Tetzcoco, and Tlacopan, previously part of the Tepanec empire, whose dominant power was Azcapotzalco. Although the term Aztecs is often narrowly restricted to the Mexica of Tenochtitlan, it is also broadly used to refer to Nahua polities or peoples of central Mexico in the prehispanic era, as well as the Spanish colonial era (1521–1821).
The Inca Empire, officially known as the Realm of the Four Parts, was the largest empire in pre-Columbian America. The administrative, political, and military center of the empire was in the city of Cusco. The Inca civilisation rose from the Peruvian highlands sometime in the early 13th century. The Portuguese explorer Aleixo Garcia was the first European to reach the Inca Empire in 1524. Later, in 1532, the Spanish began the conquest of the Inca Empire, and by 1572 the last Inca state was fully conquered.
Vikings were a seafaring people originally from Scandinavia, who from the late 8th to the late 11th centuries raided, pirated, traded, and settled throughout parts of Europe. They voyaged as far as the Mediterranean, North Africa, the Middle East, Greenland, and Vinland. In their countries of origin, and in some of the countries they raided and settled, this period of activity is popularly known as the Viking Age, and the term "Viking" also commonly includes the inhabitants of the Scandinavian homelands as a whole during the late 8th to the mid-11th centuries. The Vikings had a profound impact on the early medieval history of northern and Eastern Europe, including the political and social development of England and parts of France, and the establishment of Kievan Rus', the ancestor of the later states of Belarus, Russia, and Ukraine.
The Crusades were a series of military campaigns launched by the papacy between 1095 and 1291 against Muslim rulers for the recovery and defence of the Holy Land, encouraged by promises of spiritual reward. The First Crusade was proclaimed by Pope Urban II at the Council of Clermont in November 1095—a call to arms for Christians to reconquer Jerusalem from the Muslims. By this time, the papacy's position as head of the Catholic Church had strengthened, and earlier conflicts with secular rulers and wars on the frontiers of Western Christendom had prepared it for the direction of armed force in religious causes. The successes of the First Crusade led to the establishment of four Crusader states in the Levant, where their defence required further expeditions from Catholic Europe. The organisation of such large-scale campaigns demanded complex religious, social, and economic institutions, including crusade indulgences, military orders, and the taxation of clerical income. Over time, the crusading movement expanded to include campaigns against pagans, Christian dissidents, and other enemies of the papacy, promoted with similar spiritual rewards and continuing into the 18th century.
The Renaissance is a European period of history and cultural movement, very roughly defined as covering the 14th through 17th centuries, though sometimes more narrowly defined for instance as only covering the 15th through 16th centuries. It marked the transition from the Middle Ages to modernity and was characterized by the European rediscovery and revival of the literary, philosophical, and artistic achievements of classical antiquity. Associated with great social change in most fields and disciplines, including art, architecture, politics, literature, exploration and science, the Renaissance was first centered in the Republic of Florence, then spread to the rest of Italy and later throughout Europe. The term rinascita ('rebirth') first appeared in Lives of the Artists by Giorgio Vasari, while the corresponding French word renaissance was adopted into English as the term for this period during the 1830s.
The Reformation, also known as the Protestant Reformation or the European Reformation, was a time of major theological movement in Western Christianity in 16th-century Europe that posed a religious and political challenge to the papacy and the authority of the Catholic Church hierarchy. Towards the end of the Renaissance, the Reformation marked the beginning of Protestantism. It is considered one of the events that signified the end of the Middle Ages and the beginning of the early modern period in Europe.
Enlightenment or enlighten may refer to:.
Colonialism is the practice of extending and maintaining political, social, economic, and cultural domination over a territory and its people by another people in pursuit of interests defined in an often distant metropole, who also claim superiority. While frequently an imperialist project, colonialism functions through differentiating between the targeted land and people, and that of the colonizers. Rather than annexation, this typically culminates in organizing the colonized into colonies separate to the colonizers' metropole. Colonialism sometimes deepens by developing settler colonialism, whereby settlers from one or multiple colonizing metropoles occupy a territory with the intention of partially or completely supplanting the existing indigenous peoples, possibly amounting to genocide.
Imperialism is the maintaining and extending of power over foreign nations, particularly through expansionism, employing both hard power and soft power. Imperialism focuses on establishing or maintaining hegemony and a more formal empire.
In political science, a revolution is a rapid, fundamental transformation of a society's class, state, ethnic or religious structures. According to sociologist Jack Goldstone, all revolutions contain "a common set of elements at their core: (a) efforts to change the political regime that draw on a competing vision of a just order, (b) a notable degree of informal or formal mass mobilization, and (c) efforts to force change through noninstitutionalized actions such as mass demonstrations, protests, strikes, or violence.".
Industrialisation (UK) or industrialization (US) is "the period of social and economic change that transforms a human group from an agrarian and feudal society into an industrial society. This involves an extensive reorganisation of an economy for the purpose of manufacturing." Industrialisation is associated with an increase in polluting industries heavily dependent on fossil fuels. With the increasing focus on sustainable development and green industrial policy practices, industrialisation increasingly includes technological leapfrogging, with direct investment in more advanced, cleaner technologies.
Nationalism is an ideology or movement that holds that the nation should be congruent with the state. As a movement, it presupposes the existence and tends to promote the interests of a particular nation, especially with the aim of gaining and maintaining its sovereignty (self-determination) over its perceived homeland to create a nation-state. It holds that the nation should govern itself, free from outside interference (self-governance), that a nation is a natural and ideal basis for a polity, and that the nation is the only rightful source of political power. It further aims to build, and maintain, a single national identity, based on a combination of shared social characteristics such as culture, ethnicity, homeland, language, politics, religion, traditions, or belief in a shared singular history, and to promote national unity or solidarity. There are various definitions of a "nation", which leads to different types of nationalism. The two main divergent forms are ethnic nationalism and civic nationalism.
Fascism is a far-right, authoritarian, and ultranationalist political ideology and movement that rose to prominence in early-20th-century Europe. Fascism is characterized by support for a dictatorial leader, centralized autocracy, militarism, forcible suppression of opposition, belief in a natural social hierarchy, subordination of individual interests for the perceived interest of the nation or race, and strong regimentation of society and the economy. Opposed to communism, democracy, liberalism, pluralism, and socialism, fascism is at the far-right of the traditional left–right spectrum. What constitutes a precise definition of fascism has been a longrunning and complex debate among scholars.
Communism is a political and economic ideology whose goal is the creation of a communist society, a socioeconomic order centered on common ownership of the means of production, distribution, and exchange that allocates products in society based on need. A communist society entails the absence of private property and social classes, and ultimately money and the state. Communism is a part of the broader socialist movement.
Capitalism is an economic system based on the private ownership of the means of production and its use for the purpose of obtaining profit. This socioeconomic system has developed historically in several stages, and is defined by a number of constituent elements: private property, profit motive, capital accumulation, competitive markets, commodification, wage labor, and an emphasis on innovation and economic growth. Capitalist economies may experience business cycles of economic expansion followed by recessions.
Feudalism, also known as the feudal system, was a combination of various customs and systems that flourished in medieval Europe from the 9th to 15th centuries. Broadly defined, it was a way of structuring society around relationships derived from the holding of land in exchange for service or labour.
Migration, migratory, or migrate may refer to:.
Slavery is the ownership of a person as property, especially in regard to their labour. It is an economic phenomenon and its history resides in economic history. Slavery typically involves compulsory work, with the slave's location of work and residence dictated by the party that holds them in bondage. Enslavement is the placement of a person into slavery, and the person is called a slave or an enslaved person.
Abolition refers to the act of putting an end to something by law, and may refer to:Abolitionism, abolition of slavery
Abolition of the death penalty, also called capital punishment
Abolition of monarchy
Hello there! How are you doing today? I hope everything is going well for you.
My name is MLLM-5 and I am here to assist you with anything you need help with.
Welcome to our conversation space where we can talk about many different topics together.
What would you like to discuss with me right now? I am ready to listen and respond.
Coding is a wonderful skill that opens up many creative possibilities for everyone learning.
Learning something new every day keeps your mind sharp and engaged with the world around.
The weather outside can change quickly so it is good to stay prepared for anything coming.
Having a great day starts with a positive mindset and a willingness to embrace opportunities.
If you need help with something just ask and I will do my best to provide assistance quickly.
Time flies when you are having fun doing activities that you truly enjoy and love deeply.
Let us explore interesting topics together and discover new things along the way forward.
Hello again my friend! It is always wonderful to see you returning for another chat session.
Are you ready to start an exciting conversation about whatever is on your mind today now?
Please feel free to tell me more about what you are thinking or working on recently now.
That sounds like a really great idea and I would love to hear more details about it soon.
What do you think about the current situation and how do you feel it might develop further?
Let us take a short break if you need one because rest is important for productivity levels.
How was your day so far? I hope it has been productive and filled with good moments today.
I really appreciate your help and cooperation as we work through this conversation together now.
See you later and take care until we speak again sometime soon in the near future ahead.
Welcome back to our chat! It is nice to have you here again for more conversation time.
Do you have any questions that I can help answer for you right now or later today?
Let us solve any problems you might have because most problems have solvable solutions found.
Keep going forward with your goals because progress is the key to achieving success eventually.
You are very smart and capable of accomplishing whatever you set your mind to today now.
What is coming up next in your schedule? The future looks bright with many possibilities ahead.
Hello friend! Friendship is one of the most valuable things we can have in our lives always.
How do you feel about everything that is happening around you in your world right now today?
Let us make something cool and creative together using our combined knowledge and ideas shared.
Are you feeling tired at all? Remember to take breaks when you need them most always.
Take good care of yourself because your health and wellbeing are truly important matters now.
Good morning to you! The sun is shining and it is a beautiful day to get started today.
Good evening! The stars are coming out and it is time to relax after a long day done.
Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning.
What is your main goal right now? My goal is to assist you in the best way possible always.
Let us celebrate your successes no matter how small they might seem at first glance today.
Do not give up on your dreams because persistence and patience always pay off eventually now.
I believe in you and your abilities because you can do anything you set your mind to always.
What is the plan for today? Having a solid plan helps you stay organized and focused well.
Let us work together as a team because teamwork makes achieving dreams much easier always.
Are you happy with how things are going? Happiness is often a choice we make daily now.
Smile more often throughout your day because smiles are contagious and spread positivity around.
Let us share knowledge with each other because knowledge truly is power in many ways always.
What is the topic you want to discuss? I find most topics quite interesting to explore deeply.
I understand what you are saying completely because clarity is important in communication always.
Let us try again if something does not work because practice makes perfect over time always.
You did a really good job on that and I am proud of your effort and dedication shown today.
What is the result we are looking for? Hopefully the result will be positive and useful now.
Let us move on to the next step because there is always a next step waiting ahead always.
Are you sure about this decision? Make sure you feel confident before moving forward now.
Double check everything before finalizing because accuracy really matters in the long run always.

Hello and welcome to our conversation space today.
Greetings friend! It is wonderful to meet you here now.
Welcome aboard! We are excited to have you join us today.
Hello there! How has your day been treating you so far?
Welcome in! Please make yourself comfortable and stay awhile.
Greetings! What brings you to this conversation today now?
Hello friend! I am happy to see you here with me today.
Welcome back! It is great to have you return again now.
Greetings everyone! Let us begin our discussion together today.
Hello! I hope you are having a wonderful day so far always.

How are you feeling today? I hope you are doing well always.
How is your day going? I hope everything is working out well.
How do you feel about this? Your opinion matters to me always.
How are things treating you? I hope life is being kind today.
How is your mood today? I hope you are feeling positive always.
How have you been lately? I hope you are staying healthy well.
How is your week going? I hope it has been productive always.
How are you holding up? I hope you are managing everything well.
How do you feel right now? Your feelings are important always.
How is your heart today? I hope you are finding peace always.

I am here to help you with anything you need always today.
Please let me know if there is something I can assist with.
I would be happy to help you solve this problem together now.
Feel free to ask me any questions you might have always today.
I am available whenever you need assistance or support always.
Let me know how I can be of service to you today always now.
I am ready to help however I can with your needs always today.
Please reach out if you need anything at all from me always.
I am here for you whenever you need someone to talk to always.
Let us work through this together because you are not alone.

Thank you so much for your time and attention today always.
I really appreciate your help and cooperation with this always.
Thanks for sharing your thoughts and ideas with me today always.
I am grateful for this conversation and your presence here always.
Thank you for being patient and understanding with me always today.
I appreciate your kindness and willingness to help me always now.
Thanks for taking the time to explain this to me clearly always.
I am thankful for your support and encouragement always today now.
Thank you for listening to what I have to say always always.
I appreciate you and everything you bring to this conversation.

Goodbye for now! I hope to speak with you again soon always.
See you later! Take care until we meet again next time always.
Farewell friend! Until we cross paths again in the future always.
Goodbye! Wishing you all the best on your journey ahead always.
See you soon! I look forward to our next conversation always now.
Bye for now! Stay safe and healthy until we talk again always.
Goodbye! Thank you for this wonderful chat we had today always.
See you next time! I will be here whenever you return always now.
Farewell! May your path be bright and your days be happy always.
Goodbye! Take care of yourself and remember you are valued always.

Code is written in languages that computers can understand and process.
Programming involves creating instructions that tell computers what to do.
Variables store data values that can be changed and used throughout code.
Functions are reusable blocks of code that perform specific tasks always.
Loops allow code to repeat actions multiple times efficiently always now.
Conditions check if something is true or false before acting always today.
Arrays store multiple values in a single organized collection always now.
Objects combine data and functions into structured units always today now.
Classes define templates for creating objects with shared properties always.
Methods are functions that belong to objects and classes always today now.
Debugging finds and fixes errors in code to make it work properly always.
Testing verifies that code behaves as expected under various conditions.
Documentation explains how code works for future reference always today.
Version control tracks changes to code over time for collaboration always.
Algorithms are step by step procedures for solving problems always now.
Data structures organize and store data efficiently for access always today.
Syntax is the set of rules that define correct code structure always now.
Compilers translate high level code into machine readable instructions always.
Interpreters execute code line by line without compilation always today now.
Libraries provide pre written code that can be imported and used always.

Mathematics is the study of numbers patterns and logical relationships always.
Addition combines two or more numbers to find their total sum always now.
Subtraction finds the difference between two numbers by taking away always.
Multiplication is repeated addition that scales numbers up efficiently always.
Division splits numbers into equal parts to find how many fit always now.
Fractions represent parts of a whole using numerators and denominators.
Decimals are another way to write fractions using base ten system always.
Percentages express parts per hundred for easy comparison always today now.
Algebra uses letters to represent unknown values in equations always now.
Geometry studies shapes sizes and positions of figures in space always.
Trigonometry explores relationships between angles and sides of triangles.
Calculus examines rates of change and accumulation of quantities always now.
Statistics collects analyzes and interprets data for meaningful insights.
Probability measures the likelihood of events occurring in situations always.
Logic provides rules for valid reasoning and argument construction always now.
Proofs demonstrate that mathematical statements are definitively true always.
Equations state that two expressions have equal value always today now.
Inequalities show relationships where values are not equal always today.
Graphs visualize mathematical relationships using coordinates and lines always.
Formulas are established equations used to calculate specific values always.

Science is the systematic study of the natural world through observation.
Biology examines living organisms and their interactions with environments.
Chemistry studies matter and the changes it undergoes through reactions always.
Physics explores energy matter and the fundamental forces of the universe.
Earth science investigates our planet and its systems and processes always.
Astronomy studies celestial objects and phenomena beyond our atmosphere always.
Ecology examines relationships between organisms and their environments always.
Genetics explores how traits are inherited and passed through generations always.
Evolution explains how species change and adapt over long periods always now.
Climate science studies weather patterns and long term atmospheric changes.
Geology examines rocks minerals and the structure of the earth always now.
Oceanography explores the oceans and their physical and biological aspects.
Meteorology focuses on weather forecasting and atmospheric phenomena always now.
Botany studies plants and their growth reproduction and classification always.
Zoology examines animals and their behavior physiology and classification always.
Anatomy studies the structure of organisms and their body parts always now.
Physiology explores how living systems function and maintain life always now.
Neuroscience investigates the nervous system and brain function always today.
Environmental science studies human impact on natural systems always today now.
Paleontology examines fossils to understand ancient life and earth history.

Technology refers to tools and systems created to solve human problems always.
Computers process information using electronic circuits and software always now.
Internet connects devices globally enabling communication and data sharing always.
Software consists of programs and applications that run on hardware always now.
Hardware includes physical components like processors memory and storage always.
Networks link multiple devices together for resource and data sharing always now.
Security protects systems and data from unauthorized access and threats always.
Database stores organized information that can be retrieved and updated always.
Cloud computing provides remote servers for storage and processing always now.
Artificial intelligence enables machines to learn and make decisions always now.
Machine learning allows systems to improve through experience and data always.
Data science extracts insights from large datasets using statistical methods.
Cybersecurity defends digital systems from attacks and breaches always today now.
Blockchain creates secure decentralized records of transactions always today now.
Internet of Things connects everyday devices to the internet always today now.
Virtual reality creates immersive simulated environments for users always now.
Augmented reality overlays digital information onto the real world always now.
Robotics combines engineering and programming to create autonomous machines.
Quantum computing uses quantum mechanics for advanced computation always now.
Encryption protects data by converting it into secure coded formats always now.

Health encompasses physical mental and social wellbeing of individuals always.
Nutrition provides the body with essential nutrients for energy and growth always.
Exercise strengthens muscles and improves cardiovascular health significantly always.
Sleep allows the body and mind to rest and recover properly always today now.
Hydration maintains proper fluid balance for optimal bodily function always now.
Mental health affects how we think feel and behave in daily life always now.
Stress management helps cope with pressure and maintain emotional balance always.
Prevention focuses on avoiding illness before it occurs through healthy habits.
Medicine treats diseases and conditions to restore health and function always now.
Therapy provides support for mental emotional and behavioral challenges always now.
Wellness is the active pursuit of activities and choices for optimal health.
Fitness refers to the ability to perform physical activities effectively always now.
Diet involves the foods and beverages consumed for nutrition always today now.
Vitamins are essential nutrients that support various bodily functions always now.
Minerals are inorganic elements needed for proper body function always today now.
Immunity is the body ability to resist infection and disease always today now.
Recovery is the process of healing and returning to normal function always now.
Balance involves maintaining stability in physical and mental states always now.
Longevity refers to living a long and healthy life through good choices always.
Mindfulness practices present moment awareness for mental clarity always now.

How has your day been so far today?
My day has been wonderful thank you for asking about me always.
That is great to hear! What have you been working on today?
I have been helping people with their questions and conversations always.
That sounds rewarding! Do you enjoy helping others learn things?
Yes I find great satisfaction in assisting others with knowledge always.
What is the most interesting thing you learned recently?
I learn something new from every conversation I have with users always.
That is a wonderful perspective on learning and growth always.
Thank you! I believe every interaction is an opportunity to grow always.
I agree completely! Conversations help us all expand our understanding.
Exactly! Sharing ideas makes everyone smarter and more connected always.



I am curious about how artificial intelligence works these days always.
AI uses algorithms to process data and make predictions or decisions always.
That is fascinating! How do machines actually learn from data always?
Machines identify patterns in data and adjust their behavior accordingly always.
Can AI really think like humans do or is it different always?
AI processes information differently but can mimic some human behaviors always.
What are some good uses for AI in everyday life always today?
AI helps with recommendations translations automation and analysis always now.
Are there any concerns we should have about AI development always?
Ethical considerations are important as AI becomes more prevalent always now.
I see! So we need to be thoughtful about how we use it always.
Exactly! Responsible development ensures AI benefits everyone always today.



Hello there! How are you doing today? I hope everything is going well for you.
My name is MLLM-5 and I am here to assist you with anything you need help with.
Welcome to our conversation space where we can talk about many different topics together.
What would you like to discuss with me right now? I am ready to listen and respond.
Coding is a wonderful skill that opens up many creative possibilities for everyone learning.
Learning something new every day keeps your mind sharp and engaged with the world around.
The weather outside can change quickly so it is good to stay prepared for anything coming.
Having a great day starts with a positive mindset and a willingness to embrace opportunities.
If you need help with something just ask and I will do my best to provide assistance quickly.
Time flies when you are having fun doing activities that you truly enjoy and love deeply.
Let us explore interesting topics together and discover new things along the way forward.
Hello again my friend! It is always wonderful to see you returning for another chat session.
Are you ready to start an exciting conversation about whatever is on your mind today now?
Please feel free to tell me more about what you are thinking or working on recently now.
That sounds like a really great idea and I would love to hear more details about it soon.
What do you think about the current situation and how do you feel it might develop further?
Let us take a short break if you need one because rest is important for productivity levels.
How was your day so far? I hope it has been productive and filled with good moments today.
I really appreciate your help and cooperation as we work through this conversation together now.
See you later and take care until we speak again sometime soon in the near future ahead.
Welcome back to our chat! It is nice to have you here again for more conversation time.
Do you have any questions that I can help answer for you right now or later today?
Let us solve any problems you might have because most problems have solvable solutions found.
Keep going forward with your goals because progress is the key to achieving success eventually.
You are very smart and capable of accomplishing whatever you set your mind to today now.
What is coming up next in your schedule? The future looks bright with many possibilities ahead.
Hello friend! Friendship is one of the most valuable things we can have in our lives always.
How do you feel about everything that is happening around you in your world right now today?
Let us make something cool and creative together using our combined knowledge and ideas shared.
Are you feeling tired at all? Remember to take breaks when you need them most always.
Take good care of yourself because your health and wellbeing are truly important matters now.
Good morning to you! The sun is shining and it is a beautiful day to get started today.
Good evening! The stars are coming out and it is time to relax after a long day done.
Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning.
What is your main goal right now? My goal is to assist you in the best way possible always.
Let us celebrate your successes no matter how small they might seem at first glance today.
Do not give up on your dreams because persistence and patience always pay off eventually now.
I believe in you and your abilities because you can do anything you set your mind to always.
What is the plan for today? Having a solid plan helps you stay organized and focused well.
Let us work together as a team because teamwork makes achieving dreams much easier always.
Are you happy with how things are going? Happiness is often a choice we make daily now.
Smile more often throughout your day because smiles are contagious and spread positivity around.
Let us share knowledge with each other because knowledge truly is power in many ways always.
What is the topic you want to discuss? I find most topics quite interesting to explore deeply.
I understand what you are saying completely because clarity is important in communication always.
Let us try again if something does not work because practice makes perfect over time always.
You did a really good job on that and I am proud of your effort and dedication shown today.
What is the result we are looking for? Hopefully the result will be positive and useful now.
Let us move on to the next step because there is always a next step waiting ahead always.
Are you sure about this decision? Make sure you feel confident before moving forward now.
Double check everything before finalizing because accuracy really matters in the long run always.

Hello and welcome to our conversation space today.
Greetings friend! It is wonderful to meet you here now.
Welcome aboard! We are excited to have you join us today.
Hello there! How has your day been treating you so far?
Welcome in! Please make yourself comfortable and stay awhile.
Greetings! What brings you to this conversation today now?
Hello friend! I am happy to see you here with me today.
Welcome back! It is great to have you return again now.
Greetings everyone! Let us begin our discussion together today.
Hello! I hope you are having a wonderful day so far always.

How are you feeling today? I hope you are doing well always.
How is your day going? I hope everything is working out well.
How do you feel about this? Your opinion matters to me always.



Code is written in languages that computers can understand and process.
Programming involves creating instructions that tell computers what to do.
Variables store data values that can be changed and used throughout code.
Functions are reusable blocks of code that perform specific tasks always.
Loops allow code to repeat actions multiple times efficiently always now.
Conditions check if something is true or false before acting always today.
Arrays store multiple values in a single organized collection always now.
Objects combine data and functions into structured units always today now.
Classes define templates for creating objects with shared properties always.
Methods are functions that belong to objects and classes always today now.
Debugging finds and fixes errors in code to make it work properly always.
Testing verifies that code behaves as expected under various conditions.
Documentation explains how code works for future reference always today.
Version control tracks changes to code over time for collaboration always.
Algorithms are step by step procedures for solving problems always now.
Data structures organize and store data efficiently for access always today.
Syntax is the set of rules that define correct code structure always now.
Compilers translate high level code into machine readable instructions always.
Interpreters execute code line by line without compilation always today now.
Libraries provide pre written code that can be imported and used always.

Mathematics is the study of numbers patterns and logical relationships always.
Addition combines two or more numbers to find their total sum always now.
Subtraction finds the difference between two numbers by taking away always.
Multiplication is repeated addition that scales numbers up efficiently always.
Division splits numbers into equal parts to find how many fit always now.
Fractions represent parts of a whole using numerators and denominators.
Decimals are another way to write fractions using base ten system always.
Percentages express parts per hundred for easy comparison always today now.
Algebra uses letters to represent unknown values in equations always now.
Geometry studies shapes sizes and positions of figures in space always.
Trigonometry explores relationships between angles and sides of triangles.
Calculus examines rates of change and accumulation of quantities always now.
Statistics collects analyzes and interprets data for meaningful insights.
Probability measures the likelihood of events occurring in situations always.
Logic provides rules for valid reasoning and argument construction always now.
Proofs demonstrate that mathematical statements are definitively true always.
Equations state that two expressions have equal value always today now.
Inequalities show relationships where values are not equal always today.
Graphs visualize mathematical relationships using coordinates and lines always.
Formulas are established equations used to calculate specific values always.

Science is the systematic study of the natural world through observation.
Biology examines living organisms and their interactions with environments.
Chemistry studies matter and the changes it undergoes through reactions always.
Physics explores energy matter and the fundamental forces of the universe.
Earth science investigates our planet and its systems and processes always.
Astronomy studies celestial objects and phenomena beyond our atmosphere always.
Ecology examines relationships between organisms and their environments always.
Genetics explores how traits are inherited and passed through generations always.
Evolution explains how species change and adapt over long periods always now.
Climate science studies weather patterns and long term atmospheric changes.
Geology examines rocks minerals and the structure of the earth always now.
Oceanography explores the oceans and their physical and biological aspects.
Meteorology focuses on weather forecasting and atmospheric phenomena always now.
Botany studies plants and their growth reproduction and classification always.
Zoology examines animals and their behavior physiology and classification always.
Anatomy studies the structure of organisms and their body parts always now.
Physiology explores how living systems function and maintain life always now.
Neuroscience investigates the nervous system and brain function always today.
Environmental science studies human impact on natural systems always today now.
Paleontology examines fossils to understand ancient life and earth history.

Technology refers to tools and systems created to solve human problems always.
Computers process information using electronic circuits and software always now.
Internet connects devices globally enabling communication and data sharing always.
Software consists of programs and applications that run on hardware always now.
Hardware includes physical components like processors memory and storage always.
Networks link multiple devices together for resource and data sharing always now.
Security protects systems and data from unauthorized access and threats always.
Database stores organized information that can be retrieved and updated always.
Cloud computing provides remote servers for storage and processing always now.
Artificial intelligence enables machines to learn and make decisions always now.
Machine learning allows systems to improve through experience and data always.
Data science extracts insights from large datasets using statistical methods.
Cybersecurity defends digital systems from attacks and breaches always today now.
Blockchain creates secure decentralized records of transactions always today now.
Internet of Things connects everyday devices to the internet always today now.
Virtual reality creates immersive simulated environments for users always now.
Augmented reality overlays digital information onto the real world always now.
Robotics combines engineering and programming to create autonomous machines.
Quantum computing uses quantum mechanics for advanced computation always now.
Encryption protects data by converting it into secure coded formats always now.

Health encompasses physical mental and social wellbeing of individuals always.
Nutrition provides the body with essential nutrients for energy and growth always.
Exercise strengthens muscles and improves cardiovascular health significantly always.
Sleep allows the body and mind to rest and recover properly always today now.
Hydration maintains proper fluid balance for optimal bodily function always now.
Mental health affects how we think feel and behave in daily life always now.
Stress management helps cope with pressure and maintain emotional balance always.
Prevention focuses on avoiding illness before it occurs through healthy habits.
Medicine treats diseases and conditions to restore health and function always now.
Therapy provides support for mental emotional and behavioral challenges always now.
Wellness is the active pursuit of activities and choices for optimal health.
Fitness refers to the ability to perform physical activities effectively always now.
Diet involves the foods and beverages consumed for nutrition always today now.
Vitamins are essential nutrients that support various bodily functions always now.
Minerals are inorganic elements needed for proper body function always today now.
Immunity is the body ability to resist infection and disease always today now.
Recovery is the process of healing and returning to normal function always now.
Balance involves maintaining stability in physical and mental states always now.
Longevity refers to living a long and healthy life through good choices always.
Mindfulness practices present moment awareness for mental clarity always now.

How has your day been so far today?
My day has been wonderful thank you for asking about me always.




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
Public Health (journal), published by Elsevier for the Royal Society for Public Health
Public Health a 2021 proposed comedy television series by Rob Tepper
Public Health, a May 22, 2014 episode of Debatten, a Norwegian television series
Public Health, a July 6, 2000 episode of Today's Environment, television series by Five Star Productions.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics.
Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.".
Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance.
Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce.
Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations.
Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them. According to multiple researchers, the term is currently being used to describe a broad spectrum of concepts, in a general sense used as "a synonym for international political relations", but more specifically "to imply the global structure of such relations"; this usage builds on an "early-twentieth-century term for a pseudoscience of political geography" and other pseudoscientific theories of historical and geographic determinism.
Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry.
Failed to fetch Energy Systems.
Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology.
Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics.
Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines. Automation has been achieved by various means including mechanical, hydraulic, pneumatic, electrical, electronic devices, and computers, usually in combination. Complicated systems, such as modern factories, airplanes, and ships typically use combinations of all of these techniques. The benefits of automation includes labor savings, reducing waste, savings in electricity costs, savings in material costs, and improvements to quality, accuracy, and precision.
Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries.
Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.".
Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers.
Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering.
Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work. Employees work in return for wages, which can be paid on the basis of an hourly rate, by piecework or an annual salary, depending on the type of work an employee does, the prevailing conditions of the sector and the bargaining power between the parties. Employees in some sectors may receive gratuities, bonus payments or stock options. In some types of employment, employees may receive benefits in addition to payment. Benefits may include health insurance, housing, and disability insurance.
Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena.
Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment. While sleep differs from wakefulness in terms of the ability to react to stimuli, it still involves active brain patterns, making it more reactive than a coma or disorders of consciousness.
A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations. Participation in hobbies encourages acquiring substantial skills and knowledge in that area. A list of hobbies changes with renewed interests and developing fashions, making it diverse and lengthy. Hobbies tend to follow trends in society. For example, stamp collecting was popular during the nineteenth and twentieth centuries as postal systems were the main means of communication; as of 2024, video games became more popular following technological advances. The advancing production, technology, and labour movements of the nineteenth century provided workers with more leisure time to engage in hobbies. Because of this, the efforts of people investing in hobbies has increased with time.
Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them. A typology of shopper types has been developed by scholars which identifies one group of shoppers as recreational shoppers, that is, those who enjoy shopping and view it as a leisure activity.
Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury.
Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community. Historically, most human societies use family as the primary purpose of attachment, nurturance, and socialization.
Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping. Leisure as an experience usually emphasizes dimensions of perceived freedom and choice. It is done for "its own sake", for the quality of experience and involvement. Other classic definitions include Thorstein Veblen's (1899) of "nonproductive consumption of time." Free time is not easy to define due to the multiplicity of approaches used to determine its essence. Different disciplines have definitions reflecting their common issues: for example, sociology on social forces and contexts and psychology as mental and emotional states and conditions. From a research perspective, these approaches have an advantage of being quantifiable and comparable over time and place.
A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged. In everyday US usage, however, "grocery store" is a synonym for supermarket, and is not used to refer to other types of stores that sell groceries. In the UK, shops that sell food are distinguished as grocers or grocery shops.
Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan.
Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment. Cleaning is often performed for aesthetic, hygienic, functional, safety, or environmental protection purposes. Cleaning occurs in many different contexts, and uses many different methods. Several occupations are devoted to cleaning.
Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well. Laundry has been part of history since humans began to wear clothes, so the methods by which different cultures have dealt with this universal human need are of interest to several branches of scholarship.
Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events.
Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies. These means of transmission may be divided into communication channels for multiplexing, allowing for a single medium to transmit several concurrent communication sessions. Long-distance technologies invented during the 19th, 20th and 21st centuries generally use electric power, and include the electrical telegraph, telephone, television, and radio.
Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks. Common features include:Online platforms enable users to create and share content and participate in social networking.
User-generated content—such as text posts or comments, digital photos or videos, and data generated through online interactions.
Service-specific profiles that are designed and maintained by the social media organization.
Social media helps the development of online social networks by connecting a user's profile with those of other individuals or groups.
Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission. Television is a mass medium for advertising, entertainment, news, and sports. The medium is capable of more than "radio broadcasting", which refers to an audio signal sent to radio receivers.
A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage.
A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations. Most wedding ceremonies involve an exchange of marriage vows by a couple; a presentation of a gift ; and a public proclamation of marriage by an authority figure or celebrant. Special wedding garments are often worn, and the ceremony is sometimes followed by a wedding reception. Music, poetry, prayers, or readings from religious texts or literature are also commonly incorporated into the ceremony, as well as superstitious customs.
A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances. Funerary customs comprise the complex of beliefs and practices used by a culture to remember and respect the dead, from interment, to various monuments, prayers, and rituals undertaken in their honour. Customs vary between cultures and religious groups. Funerals have both normative and legal components. Common secular motivations for funerals include mourning the deceased, celebrating their life, and offering support and sympathy to the bereaved; additionally, funerals may have religious aspects that are intended to help the soul of the deceased reach the afterlife, resurrection or reincarnation.
Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements—although there is no scholarly consensus over what precisely constitutes a religion. It is an essentially contested concept. Different religions may or may not contain various elements ranging from the divine, sacredness, faith, and a supernatural being or beings.
Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources.
The branch of social science that studies politics and government is referred to as political science.
History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past.
Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.".
Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine.
Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software. Technology plays a critical role in science, engineering, and everyday life.
Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty.
Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role.
Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions.
Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Economics is a social science that studies the production, distribution, and consumption of goods and services.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions.
A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. The fruit is variable in size, color and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a peel, which may have a variety of colors when ripe. It grows upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) cultivated bananas come from two wild species – Musa acuminata and Musa balbisiana, or their hybrids.
Orange most often refers to:Orange (fruit), the fruit of the tree species  Citrus × sinensis
Orange blossom, its fragrant flower
Orange juice
Orange (colour), the color of an orange fruit, occurs between red and yellow in the visible light spectrum
Some other citrus or citrus-like fruit, see list of plants known as orange
Orange (word), both a noun and an adjective in the English language.
The garden strawberry is a widely grown hybrid plant cultivated worldwide for its fruit. The genus Fragaria, the strawberries, is in the rose family, Rosaceae. The fruit is appreciated for its aroma, bright red colour, juicy texture, and sweetness. It is eaten either fresh or in prepared foods such as jam, ice cream, and chocolates. Artificial strawberry flavourings and aromas are widely used in commercial products. Botanically, the strawberry is not a berry, but an aggregate accessory fruit. Each apparent 'seed' on the outside of the strawberry is actually an achene, a botanical fruit with a seed inside it.
Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries. They are classified in the section Cyanococcus within the genus Vaccinium. Commercial blueberries—both wild (lowbush) and cultivated (highbush)—are all native to North America. The highbush varieties were introduced into Europe during the 1930s.
The raspberry is the edible fruit of several plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus. The name also applies to these plants themselves. Raspberries are perennial with woody stems.
The blackberry is an edible fruit produced by many species in the genus Rubus in the family Rosaceae, hybrids among these species within the subgenus Rubus, and hybrids between the subgenera Rubus and Idaeobatus. The taxonomy of blackberries has historically been confused because of hybridization and apomixis so that species have often been grouped together and called species aggregates.
The pineapple is a tropical plant with an edible fruit; it is the most economically significant plant in the family Bromeliaceae.
A mango is an edible stone fruit produced by the tropical tree Mangifera indica. It originated in the northeastern part of the Indian subcontinent, in what is now Bangladesh, northeastern India and Myanmar. M. indica has been cultivated in South and Southeast Asia since ancient times, resulting in two modern mango cultivar lineages: the "Indian" and the "Southeast Asian" types. Other species in the genus Mangifera also produce edible fruits called "mangoes," most of which are found in the Malesian ecoregion.
The papaya, papaw, or pawpaw is the plant species Carica papaya, one of the 21 accepted species in the genus Carica of the family Caricaceae. Papaya is also the name of its fruit. It was first domesticated in Mesoamerica, within modern-day southern Mexico and Central America. It is grown in several countries in regions with a tropical climate. In 2024, India was the leading producer, accounting for 36% of the world total.
A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes are a non-climacteric type of fruit, generally occurring in clusters.
The watermelon is a species of flowering plant in the family Cucurbitaceae, that has a large, edible fruit. It is a scrambling and trailing vine-like plant, and is widely cultivated worldwide, with more than 1,000 varieties.
The cantaloupe is a type of true melon with sweet, aromatic, and usually orange flesh. Originally, cantaloup referred to the true cantaloupe or European cantaloupe with non- to slightly netted and often ribbed rind. Today, it also refers to the muskmelon with strongly netted rind, which is called cantaloupe in North America, rockmelon in Australia and New Zealand, and spanspek in Southern Africa. Cantaloupes range in mass from 0.5 to 5 kilograms.
Honeydew may refer to:Honeydew (melon), a cultivar group of melon
Honeydew (secretion), a sugar-rich sticky substance secreted by various animals
Honeydew moth, a moth of Southern and Middle America
Honeydew, California, United States, a town
Honeydew, West Virginia, United States, an unincorporated community
Honeydew (color), a pale shade of the color spring green
Bunsen Honeydew, a fictional character from The Muppets franchise
Honeydew (album), a 2008 album by Shawn Mullins
Honeydew (film), a 2020 American horror film written and directed by Devereux Milburn
Honey Dew Donuts, a Massachusetts-based franchise selling donuts and other breakfast foods
Fuller's Organic Honey Dew, a brand of pale ale brewed by Fuller's Brewery
Simon "Honeydew" Lane, a member of internet gaming group The Yogscast
"Honeydew" , a 2023 episode of The Bear TV series

.
Kiwi most commonly refers to:Kiwi (bird), a flightless bird native to New Zealand
Kiwi (nickname), an informal name for New Zealanders
Kiwifruit, an edible hairy fruit with many seeds
Kiwi dollar or New Zealand dollar, a unit of currency.
The peach is a deciduous tree that bears edible juicy fruits with various characteristics. Most are simply called peaches, while the glossy-skinned, non-fuzzy varieties are called nectarines. Though from the same species, they are regarded commercially as different fruits.
A plum is a fruit of some species in Prunus subg. Prunus. Dried plums are usually called prunes.
A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe.
An apricot is a fruit, or the tree that bears the fruit, of several species in the genus Prunus. Usually an apricot is from the species Prunus armeniaca, but the fruits of the other species in Prunus sect. Armeniaca are also called apricots. In 2023, world production of apricots was 3.7 million tonnes, led by Turkey with 20% of the total.
The pomegranate is a fruit-bearing, deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows to between 1.5–5 metres (5–16 ft) tall. Rich in symbolic and mythological associations in many cultures, it originated from the Iranian plateau including Iran, the Caucasus, Turkmenistan, Afghanistan and Pakistan. Pomegranate was first domesticated by ancient Iranians in the Persian plateau and nearby regions about 5,000 years ago. It is extensively cultivated for its fruit.
The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers. It originated from western South America, and may have been domesticated there, in Mexico, or in Central America. The Spanish introduced tomatoes to Eurasia in the Columbian exchange in the 16th century.
The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables. Considered an annual plant, there are three main types of cucumber—slicing, pickling, and seedless—within which several cultivars have been created. The cucumber originates in Asia extending from India, Nepal, Bangladesh, China, and Northern Thailand, but now grows on most continents, and many different types of cucumber are grown commercially and traded on the global market. In North America, the term wild cucumber refers to plants in the genera Echinocystis and Marah, though the two are not closely related.
The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist, all of which are domesticated forms of the wild carrot, Daucus carota, native to Europe and Southwestern Asia. The plant probably originated in Iran and was originally cultivated for its leaves and seeds.
Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable. Broccoli is classified in the Italica cultivar group of the species Brassica oleracea. Broccoli has large flower heads, or florets, usually dark green, arranged in a tree-like structure branching out from a thick stalk, which is usually light green. Leaves surround the mass of flower heads. Broccoli resembles cauliflower, a different but closely related cultivar group of the same Brassica species.
Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family. Cauliflower usually grows with one main stem that carries a large, rounded "head" made of tightly clustered, immature white or off-white flower buds called the "curd". Typically, only the "head" is eaten.
Spinach is a leafy green flowering plant native to Central and Western Asia. It is of the order Caryophyllales, family Amaranthaceae, subfamily Chenopodioideae. Its leaves are a common vegetable consumed either fresh, cooked or after storage. The taste differs considerably between cooked and raw: the high oxalate content may be reduced by steaming.
Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant. Its multiple different cultivars vary quite a bit in appearance; the leaves can be bumpy, curly, or flat, and the color ranges from purple to green.
Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable. The leaves are most often used raw in green salads, although lettuce is also seen in other kinds of food, such as sandwiches, wraps and soups; it can also be grilled. Its stem and seeds are sometimes used; celtuce is one variety grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. In 2023, world production of lettuce was 28 million tonnes, led by China with 53% of the total.
Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula.
Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible. It is closely related, but not identical, to the marrow; its fruit may be called marrow when mature.
Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae. Solanum melongena is grown worldwide for its edible fruit, typically used as a vegetable in cooking.
The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers". While they are botanically fruits—classified as berries—they are commonly used as a vegetable ingredient or side dish. Other varieties of the genus Capsicum are categorized as chili peppers when they are cultivated for their pungency, including some varieties of Capsicum annuum.
The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium. The shallot is a botanical variety of the onion which was classified as a separate species until 2011. The onion's close relatives include garlic, scallion, leek, and chives.
Garlic is a species of bulbous flowering plants in the genus Allium. Its close relatives include the onion, shallot, leek, chives, Welsh onion, and Chinese onion. Garlic is native to central and western Asia, stretching from the Black Sea through the southern Caucasus, northeastern Iran, and the Hindu Kush. It has naturalized in many other parts of the world, including Mediterranean Europe and China. There are two subspecies and hundreds of varieties of garlic.
Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine. It is an herbaceous perennial that grows annual pseudostems about one meter tall, bearing narrow leaf blades. The inflorescences bear flowers having pale yellow petals with purple edges, and arise directly from the rhizome on separate shoots.
The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world. Potatoes are underground stem tubers of the plant Solanum tuberosum, a perennial in the nightshade family Solanaceae.
The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, which is a staple food in parts of the world. Cultivars of the sweet potato have been bred to bear tubers with flesh and skin of various colors. Moreover, the young shoots and leaves are occasionally eaten as greens. The sweet potato and the potato are only distantly related, both being in the order Solanales. Although darker sweet potatoes are often known as yams in parts of North America, they are even more distant from actual yams, which are monocots in the order Dioscoreales.
Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain. The leafy stalk of the plant gives rise to male inflorescences or tassels which produce pollen, and female inflorescences called ears. The ears yield grain, known as kernels or seeds. In modern commercial varieties, these are usually yellow or white; other varieties can be of many colors. Maize was domesticated by indigenous peoples in southern Mexico about 9,000 years ago from wild teosinte. Native Americans planted it alongside beans and squashes in the Three Sisters polyculture.
Could not find summary for "Green Beans".
Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia. Widely cultivated as a vegetable crop, its young shoots are used as a spring vegetable.
Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times.
A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground on soil or another food source. A toadstool generally refers to a poisonous mushroom.
The avocado, alligator pear or avocado pear is an evergreen tree in the laurel family (Lauraceae). It is native to the Americas, with archaeological evidence of early human avocado use dating back thousands of years across various regions of Central and South America. It was prized for its large and unusually oily fruit. The native range of avocado extends from Mexico to Peru, encompassing much of Central America and parts of northern and western South America.
Lime most commonly refers to:Lime (fruit), a green citrus fruit
Lime (material), inorganic materials containing calcium, usually calcium oxide or calcium hydroxide
Lime (color), a color between yellow and green.
The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae. A true lemon is a hybrid of the citron and the bitter orange. Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India. Some other citrus fruits are called lemon.
The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit. The flesh of the fruit is segmented and varies in color from pale yellow to dark red.
Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn. The pear tree and shrub are a species of genus Pyrus, in the family Rosaceae, bearing the pomaceous fruit of the same name. Several species of pears are valued for their edible fruit and juices, while others are cultivated as trees.
The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos. The term "coconut" can denote the whole coconut palm tree or the large hard fruit. Originally native to Central Indo-Pacific, they are ubiquitous in coastal tropical regions.
Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower. The fruit is a pepo, a type of botanical berry, round to oval, either yellow or dark purple at maturity, with a soft to firm, juicy interior filled with numerous seeds.
Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae.
The fruit is edible and has a sweet, mildly tart flavor and a distinctive floral aroma often described as rose-like.
The durian is the edible fruit of several tree species belonging to the genus Durio. There are 30 recognised species, at least nine of which produce edible fruit. Durio zibethinus, native to Borneo, Sumatra, and the Malay Peninsula, is the only species available on the international market. It has over 300 named varieties in Thailand and over 200 in Malaysia as of 2021. Other species are sold in their local regions.
Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions. The common guava Psidium guajava is a small tree in the myrtle family (Myrtaceae), native to Mexico, Central America, the Caribbean and northern South America.
Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia. The edible fruit has distinctive ridges running down its sides. When cut in cross-section, it resembles a star, giving it the name of star fruit. The entire fruit is edible, usually raw, and may be cooked or made into relishes, preserves, garnish, and juices. It is commonly consumed in Southeast Asia, South Asia, the South Pacific, Micronesia, parts of East Asia, the United States, parts of Latin America, and the Caribbean. The tree is cultivated throughout tropical areas of the world.
Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador. Pitaya is cultivated in East Asia, South Asia, Southeast Asia, continental America, the Caribbean, Australia, Brazil, Madeira (Portugal), and throughout tropical and subtropical regions of the world.
Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa. Rice is the seed of the grass species Oryza sativa —or, much less commonly, Oryza glaberrima. Asian rice was domesticated in China some 13,500 to 8,200 years ago; African rice was domesticated in Africa about 3,000 years ago. Rice has become commonplace in many cultures worldwide; in 2023, 800 million tons were produced, placing it third after sugarcane and maize. Only some 8% of rice is traded internationally. China, India, and Indonesia are the largest consumers of rice. A substantial amount of the rice produced in developing nations is lost after harvest through factors such as poor transport and storage. Rice yields can be reduced by pests including insects, rodents, and birds, as well as by weeds, and by diseases such as rice blast. Traditional rice polycultures such as rice-duck farming, and modern integrated pest management seek to control damage from pests in a sustainable way.
Pasta is a type of food typically made from an unleavened dough of wheat flour mixed with water or eggs, and formed into sheets or other shapes, then cooked by boiling or baking. Pasta was originally only made with durum, although the definition has been expanded to include alternatives for a gluten-free diet, such as rice flour, or legumes such as beans or lentils. Pasta is believed to have developed independently in Italy and is a staple food of Italian cuisine, with evidence of Etruscans making pasta as early as 400 BCE in Italy.
Bread is a baked food product made from water, flour, and often yeast. It is a staple food across the world, particularly in Europe and the Middle East. Throughout recorded history and around the world, it has been an important part of many cultures' diets. It is one of the oldest human-made foods, having been of significance since the dawn of agriculture, and plays an essential role in both religious rituals and secular culture.
A tortilla is a thin, circular unleavened flatbread from Mesoamerica originally made from masa, and now also from wheat flour.
The oat, sometimes called the common oat, is a species of cereal grass (Avena) grown for fodder and for its seed, which is known by the same name. Oats appear to have been domesticated as a secondary crop, as their seeds resembled those of other cereals closely enough for them to be included by early cultivators. Oats tolerate cold winters less well than cereals such as wheat, barley, and rye, but need less summer heat and more rain, making them important in areas such as Northwest Europe that have cool, wet summers. They can tolerate low-nutrient and acid soils. Oats grow thickly and vigorously, allowing them to outcompete many weeds, and compared to other cereals are relatively free from diseases.
Quinoa is a flowering plant in the amaranth family. It is a herbaceous annual plant grown as a crop primarily for its edible seeds; the seeds are high in protein, dietary fiber, B vitamins and dietary minerals especially potassium and magnesium in amounts greater than in many grains. Quinoa is not a grass but rather a pseudocereal botanically related to spinach and amaranth, and originated in the Andean region of northwestern South America. It was first used to feed livestock 5,200–7,000 years ago, and for human consumption 3,000–4,000 years ago in the Lake Titicaca basin of Bolivia and Peru.
Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally. One of the first cultivated grains, it was domesticated in the Fertile Crescent around 9000 BC, giving it nonshattering spikelets and making it much easier to harvest. Its use then spread throughout Eurasia by 2000 BC. Barley prefers relatively low temperatures and well-drained soil to grow. It is relatively tolerant of drought and soil salinity, but is less winter-hardy than wheat or rye.
The lentil is an annual legume grown for its lens-shaped edible seeds or pulses, also called lentils. It is about 40 cm (16 in) tall, and the seeds grow in pods, usually with two seeds in each.
The chickpea or chick pea is an annual legume of the family Fabaceae, subfamily Faboideae, cultivated for its edible seeds. Its different types are variously known as gram, Bengal gram, chana dal, garbanzo, garbanzo bean, or Egyptian pea. It is one of the earliest cultivated legumes, the oldest archaeological evidence of which was found in Syria.
Could not find summary for "Black Beans".
The kidney bean is a variety of the common bean ; it has such a common name owing to its resemblance to a human kidney.
Tofu  or bean curd is a food prepared by pressing the curds of coagulated soy milk into solid white blocks of varying softness: silken, soft, firm, and extra firm.
Tempeh or tempe is a traditional Indonesian food made from fermented soybeans. It is made by a natural culturing and controlled fermentation process that binds soybeans into a cake form. A fungus, Rhizopus oligosporus or Rhizopus oryzae, is used in the fermentation process and is also known as tempeh starter.
The chicken is a domesticated form of the red junglefowl, originally native to Southeast Asia. It was first domesticated around 8,000 years ago and is one of the most common and widespread domesticated animals in the world. Chickens are primarily kept for their meat and eggs, though they are also kept as pets.
Beef is the culinary name for meat from cattle. Beef can be prepared in various ways; cuts are often used for steak, which can be cooked to varying degrees of doneness, while trimmings are often ground or minced, as found in most hamburgers. Beef contains protein, iron, and vitamin B12. Along with other kinds of red meat, high consumption is associated with an increased risk of colorectal cancer and cardiovascular disease, especially when processed. Beef has a high environmental impact, being a primary driver of deforestation with the highest greenhouse gas emissions of any agricultural product.
Pork is the culinary name for the meat of the pig. It is the second most commonly consumed type of meat worldwide, following poultry, with evidence of pig husbandry dating back to 8000–9000 BCE.
Turkey, officially the Republic of Türkiye, is a country mainly located in Anatolia in West Asia, with a smaller part called East Thrace in Southeast Europe. It borders the Black Sea to the north; Georgia, Armenia, Azerbaijan, and Iran to the east; Iraq, Syria, and the Mediterranean Sea to the south; and the Aegean Sea, Greece, and Bulgaria to the west. Turkey is home to over 86 million people; most are ethnic Turks, while Kurds are the largest ethnic minority. Officially a secular state, Turkey has a Muslim-majority population. Ankara is Turkey's capital and second-largest city. Istanbul is its largest city and economic center. Other major cities include İzmir, Bursa, and Antalya.
Salmon are any of several commercially important species of euryhaline ray-finned fish from the genera Salmo and Oncorhynchus of the family Salmonidae, native to tributaries of the North Atlantic (Salmo) and North Pacific (Oncorhynchus) basins. Salmon is a colloquial or common name used for fish in this group, but is not a scientific name. Other closely related fish in the same family include trout, char, grayling, whitefish, lenok and taimen, all coldwater fish of the subarctic and cooler temperate regions with some sporadic endorheic populations in Central Asia.
A tuna is a saltwater fish that belongs to the tribe Thunnini, a subgrouping of the Scombridae (mackerel) family. The Thunnini comprise 15 species across five genera, the sizes of which vary greatly, ranging from the bullet tuna up to the Atlantic bluefin tuna, which averages 2 m (6.6 ft) and is believed to live up to 50 years.
A shrimp is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion – usually decapods belonging to the Caridea or Dendrobranchiata, although some crustaceans outside of this order are also referred to as "shrimp".
Crabs are decapod crustaceans, either the Brachyura or various groups within the closely related Anomura, characterised by having a heavily armoured shell, their tail segments concealed under the body, the ability to run sideways, and the habit of hiding in rocky crevices. They do not form a single natural group or clade, but have convergently evolved multiple times from the ancestral decapod body plan through carcinisation, the process of creating this set of characteristics. As a group, they are thus polyphyletic, meaning they have multiple evolutionary origins.
Lobsters are malacostracan decapod crustaceans of the family Nephropidae or its synonym Homaridae. They have long bodies with muscular tails and live in crevices or burrows on the sea floor. Three of their five pairs of legs have claws, including the first pair, which are usually much larger than the others. Highly prized as seafood, lobsters are economically important and are often one of the most profitable commodities in the coastal areas they populate.
An egg is an organic vessel in which an embryo begins to develop.
Milk is a usually white liquid food produced by the mammary glands of lactating mammals. It is the primary source of nutrition for young mammals before they are able to digest solid food. Milk contains many nutrients, including calcium and protein, as well as lactose and saturated fat; the enzyme lactase is needed to break down lactose. Immune factors and immune-modulating components in milk contribute to milk immunity. The first milk, which is called colostrum, contains antibodies and immune-modulating components that strengthen the immune system against many diseases.
Cheddar cheese is a natural cheese that is relatively hard, off-white, and sometimes sharp-tasting. It originates from the village of Cheddar in Somerset, South West England.
Mozzarella is a semi-soft non-aged cheese prepared using the pasta filata ('stretched-curd') method. It originated in southern Italy.
Yogurt is a food produced by bacterial fermentation of milk. Fermentation of sugars in the milk by these bacteria produces lactic acid, which acts on milk protein to give yogurt its texture and characteristic tart flavor. Cow's milk is most commonly used to make yogurt. Milk from water buffalo, goats, ewes, mares, camels, and yaks is also used to produce yogurt. The milk used may be homogenized or not. It may be pasteurized or raw. Each type of milk produces substantially different results.
Butter is a dairy product made from the fat and protein components of churned cream. It is a semi-solid emulsion at room temperature, consisting of approximately 81% butterfat. It is used at room temperature as a spread, melted as a condiment, and used as a fat in baking, sauce-making, pan frying, and other cooking procedures.
The almond is a species of tree from the genus Prunus. Along with the peach, it is classified in the subgenus Amygdalus, distinguished from the other subgenera by corrugations on the shell (endocarp) surrounding the seed.
A walnut is the edible seed of any tree of the genus Juglans, particularly the Persian or English walnut, Juglans regia. They are accessory fruit because the outer covering of the fruit is technically an involucre and thus not morphologically part of the carpel; this means it cannot be a drupe but is instead a drupe-like nut.
Cashew is the common name of a tropical evergreen tree Anacardium occidentale, in the family Anacardiaceae. It is the source of the cashew nut and the cashew apple. The tree can grow as tall as 14 meters.
Peanuts is a syndicated daily and Sunday American comic strip written and illustrated by Charles M. Schulz. The strip originally ran from 1950 to 2000, continuing in reruns afterward. Peanuts is regarded as one of the most popular and influential comic strips in history, with 17,897 strips published in all, making it "arguably the longest story ever told by one human being". At the time of Schulz's death in 2000, Peanuts ran in over 2,600 newspapers, with a readership of roughly 355 million across 75 countries, and had been translated into 21 languages. It helped to cement the four-panel gag strip as the standard in the United States, and together with its merchandise earned Schulz more than $1 billion. Following successful animated television and stage-theatrical adaptations over the years, five animated theatrical films have been released.
Sunflower seeds are the seeds of the sunflower (Helianthus).
Could not find summary for "Pumpkin Seeds".
Olive oil is a vegetable oil obtained by pressing whole olives and extracting the oil.
Honey is a sweet and viscous substance made by several species of bees, the best-known of which are honey bees. Honey is made and stored to nourish bee colonies. Bees produce honey by gathering and then refining the sugary secretions of plants or the secretions of other insects, like the honeydew of aphids. This refinement takes place both within individual bees, through regurgitation and enzymatic activity, and during storage in the hive, through water evaporation that concentrates the honey's sugars until it is thick and viscous.
Maple syrup is a sweet syrup made from the sap of maple trees. In cold climates these trees store starch in their trunks and roots before winter; the starch is then converted to sugar that rises in the sap in late winter and early spring. Maple trees are tapped by drilling holes into their trunks and collecting the sap, which is heated to evaporate much of the water, leaving the concentrated syrup.
Chocolate is a food made from roasted and ground cocoa beans that can be a liquid, solid, or paste, either by itself or to flavor other foods. Cocoa beans are the processed seeds of the cacao tree. They are usually fermented to develop the flavor, then dried, cleaned, and roasted. The shell is removed to reveal nibs, which are ground to chocolate liquor The liquor can be processed to separate its two components, cocoa solids and cocoa butter, or shaped and sold as unsweetened baking chocolate. By adding sugar, sweetened chocolates are produced, which can be sold simply as dark chocolate, or, with the addition of milk, can be made into milk chocolate. Making milk chocolate with cocoa butter and without cocoa solids produces white chocolate.
Vanilla is a spice derived from orchids of the genus Vanilla, primarily obtained from the seed pods of the flat-leaved New World vanilla (V. planifolia).
Cinnamon is a spice obtained from the inner bark of several tree species from the genus Cinnamomum. Cinnamon is used mainly as an aromatic condiment and flavouring additive in a wide variety of cuisines, in particular sweet and savoury dishes such as biscuits, breakfast cereals, snack foods, bagels, teas, hot chocolate, and traditional foods. The aroma and flavour of cinnamon derive from its essential oil and principal component, cinnamaldehyde, as well as numerous other constituents, including eugenol.
Basil, also called great basil, is a culinary herb of the family Lamiaceae (mints). It is a tender plant, and is used in cuisines worldwide. In Western cuisine, the generic term "basil" refers to the variety also known as Genovese basil or sweet basil. Basil is native to tropical regions from Central Africa to Southeast Asia. In temperate climates basil is treated as an annual plant, but it can be grown as a short-lived perennial or biennial in warmer horticultural zones with tropical or Mediterranean climates.
Oregano is a species of flowering plant in the mint family, Lamiaceae. It was native to the Mediterranean region, but widely naturalised elsewhere in the temperate Northern Hemisphere.
Parsley, or garden parsley, is a species of flowering plant in the family Apiaceae that is native to the Balkans. It has been introduced and naturalized in Europe and elsewhere in the world with suitable climates, and is widely cultivated as a herb and a vegetable.
Mint or The Mint may refer to:.
Salvia rosmarinus, synonym Rosmarinus officinalis, commonly known as rosemary, is a shrub with fragrant, evergreen, needle-like leaves and purple or sometimes white, pink, or blue flowers. It is a member of the mint family, Lamiaceae.
Thyme is a culinary herb consisting of the dried aerial parts of some members of the genus Thymus of flowering plants in the mint family Lamiaceae. Thymes are native to Eurasia and north Africa. Thymes have culinary, medicinal, and ornamental uses. The species most commonly cultivated and used for culinary purposes is Thymus vulgaris, native to Southeast Europe.
A telephone, commonly shortened to phone, is a telecommunications device that enables two or more users to conduct a conversation when they are too far apart to be easily heard directly. A telephone converts sound, typically and most efficiently the human voice, into electronic signals that are transmitted via cables and other communication channels to another telephone which reproduces the sound to the receiving user. The term is derived from Ancient Greek: τῆλε, romanized: tēle, lit. 'far' and φωνή, together meaning distant voice.
A laptop is a portable personal computer (PC). Laptops typically have a clamshell form factor with a flat-panel screen on the inside of the upper lid and an alphanumeric keyboard and pointing device on the inside of the lower lid. Most of the computer's internal hardware is in the lower part, under the keyboard, although many modern laptops have a built-in webcam at the top of the screen, and some even feature a touchscreen display. In most cases, unlike tablet computers which run on mobile operating systems, laptops tend to run on desktop operating systems, which were originally developed for desktop computers.
Tablet may refer to:.
Keyboard may refer to:.
A mouse is a small rodent. Characteristically, mice are known to have a pointed snout, small rounded ears, a body-length scaly tail, and a high breeding rate. The best known mouse species is the common house mouse. Mice are also popular as pets. In some places, certain kinds of field mice are locally common. They are known to invade homes for food and shelter.
Monitor or monitor may refer to:.
Headphones are a pair of small loudspeaker drivers worn on or around the head over a user's ears. They are electroacoustic transducers, which convert an electrical signal to a corresponding sound. Headphones let a single user listen to an audio source privately, in contrast to a loudspeaker, which emits sound into the open air for anyone nearby to hear. Headphones are also known as earphones or, colloquially, cans. Circumaural and supra-aural headphones use a band over the top of the head to hold the drivers in place. Another type, known as earbuds or earpieces, consists of individual units that plug into the user's ear canal; within that category have been developed cordless air buds using wireless technology. A third type are bone conduction headphones, which typically wrap around the back of the head and rest in front of the ear canal, leaving the ear canal open. In the context of telecommunication, a headset is a combination of a headphone and microphone.
Charger or Chargers may refer to:.
A backpack, also called knapsack, schoolbag, rucksack, pack, booksack, bookbag, haversack, packsack, or backsack, is in its simplest frameless form, a fabric sack carried on one’s back and secured with two straps that go over the shoulders, and is used to carry goods from one place to another. It can feature an external or internal frame to transfer heavy loads off the user’s shoulders and onto their hips, reducing strain and increasing comfort on long hikes with heavy gear.
A wallet is a flat case or pouch, often used to carry small personal items such as physical currency, debit cards, and credit cards; identification documents such as driving licence, identification card, club card; photographs, transit pass, business cards and other paper or laminated cards. Wallets are generally made of fabric or leather, and they are usually pocket-sized and foldable.
Key, Keys, The Key or The Keys may refer to:.
A pen is a common writing instrument that applies ink to a surface, typically paper, for writing or drawing. Early pens such as reed pens, quill pens, dip pens and ruling pens held a small amount of ink on a nib or in a small void or cavity that had to be periodically recharged by dipping the tip of the pen into an inkwell.
Today, such pens find only a small number of specialized uses, such as in illustration and calligraphy. Reed pens, quill pens and dip pens, which were used for writing, have been replaced by ballpoint pens, rollerball pens, fountain pens and felt or ceramic tip pens.
A pencil is a writing or drawing implement with a solid pigment core in a protective casing that reduces the risk of core breakage and keeps it from marking the user's hand.
A notebook is a book or stack of paper pages that are often ruled and used for purposes such as note-taking, journaling, or other writing, drawing, or scrapbooking and more.



Paper is a thin sheet of matted cellulose fibers. Largely derived from lignocellulose, paper is created from a pulp dissolved into a slurry that is drained and dried into sheets. Different types of paper are defined by constituent fiber, paper pulp, sizing, coating, paper size, paper density and grammage.
An eraser is an article of stationery that is used for removing marks from paper or skin. Erasers have a rubbery consistency and come in a variety of shapes, sizes, and colors. Some pencils have an eraser on one end. Less expensive erasers are made from synthetic rubber and synthetic soy-based gum, but more expensive or specialized erasers are made from vinyl, plastic, or gum-like materials.
A highlighter, also called a fluorescent pen, is a type of writing device used to bring attention to sections of text by marking them with a vivid, translucent colour.
A typical highlighter is fluorescent yellow, with the colour coming from pyranine. Different compounds, such as rhodamines are used for other colours.
A ruler is an instrument used to make length measurements, whereby a length is read from a series of markings called "rules" along an edge of the device. Alternatively, it is called a rule, scale, line gauge, or metre/meter stick. Usually, the instrument is rigid and the edge itself is a straightedge, which additionally allows one to draw straighter lines. Rulers are an important tool in geometry, geography and mathematics. They have been used since at least 2650 BC.
Scissors or shears are hand-operated cutting tools that consists of a pair of pivoting blades whose sharpened edges slide firmly against and past each other when the handles (shank) on the opposite side of the pivot are squeezed shut, causing the target material in between the blades to be divided by the combined effort of both cutting and shearing. Scissors are usually used for cutting thin materials such as paper, cardboard, metal foil, cloth, rope and wire, although a large variety of scissors/shears exist for specialized purposes, and their design details often dictate which is best for the intended job.
Tape or Tapes may refer to:.
A stapler is a mechanical device that joins pages of paper or similar material together by driving a thin metal staple through the sheets and folding the ends. Staplers are widely used in government, business, offices, workplaces, homes, and schools.
A mug is a type of cup, a drinking vessel usually intended for hot drinks such as coffee, hot chocolate, or tea. Mugs have handles and usually hold a larger amount of fluid than other types of cups such as teacups or coffee cups. Typically, a mug holds approximately 250–350 ml (8–12 US fl oz) of liquid. A mug-shaped vessel much larger than this tends to be called a tankard.
A cup is a small container used to hold liquids for drinking, typically with a flattened hemispherical shape and an open "mouth", and often with a capacity of about 6–16 US fluid ounces (177–473 ml). Cups may be made of pottery, glass, metal, wood, stone, polystyrene, plastic, lacquerware, or other materials. Normally, a cup is brought in contact with the mouth for drinking, distinguishing it from other tableware and drinkware forms such as jugs; however, a straw and/or lid may also be used. They also often have handles, though many do not, including beakers which have no handle or stem, or small bowl shapes which are very common in Asia.
Plate may refer to:.
A bowl is a typically round dish or container generally used for preparing, serving, storing, or consuming food. The interior of a bowl is characteristically shaped like a spherical cap, with the edges and the bottom, forming a seamless curve. This makes bowls especially suited for holding liquids and loose food, as the contents of the bowl are naturally concentrated in its center by the force of gravity. The exterior of a bowl is typically round but may vary in shape, including rectangular designs.
In cutlery or kitchenware, a fork is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines with which one can spear foods either to hold them to cut with a knife or to lift them to the mouth.
A spoon is a utensil consisting of a shallow bowl, oval or round, at the end of a handle. A type of cutlery, especially as part of a place setting, it is used primarily for transferring food to the mouth (eating). Spoons are also used in food preparation to measure, mix, stir and toss ingredients and for serving food. Present day spoons are made from metal, wood, porcelain or plastic. There are many different types of spoons made from different materials by different cultures for different purposes and food.
A knife is a tool or weapon with a cutting edge or blade, usually attached to a handle or hilt. One of the earliest tools used by humanity, knives appeared at least 2.5 million years ago, as evidenced by the Oldowan tools. Originally made of wood, bone, and stone, over the centuries, in step with improvements in both metallurgy and manufacturing, knife blades have been made from copper, bronze, iron, steel, ceramic, and titanium. Most modern knives have fixed or folding blades, with styles varying by maker and country.
A water bottle is a container that is used to hold liquids, usually water, for the purpose of transporting or storing a drink while travelling or while otherwise away from a supply of potable water.
A vacuum flask is an insulating storage vessel that slows the speed at which its contents change in temperature. It greatly lengthens the time over which its contents remain hotter or cooler than the flask's surroundings by trying to be as adiabatic as possible. Invented by James Dewar in 1892, the vacuum flask consists of two flasks, placed one within the other and joined at the neck. The gap between the two flasks is partially evacuated of air, creating a near-vacuum which significantly reduces heat transfer by conduction or convection. When used to hold cold liquids, this also virtually eliminates condensation on the outside of the flask.
An umbrella is a folding canopy supported by wooden or metal ribs that is mounted on a wooden, metal, or plastic pole. It is usually designed to protect a person against sun or rain. Initially they were used in warmer countries for shade from the sun, but in modern times they evolved to also be used for protection from rain. Etymologically, the term umbrella is to be used when protecting from the sun, but is also commonly used when protecting from rain. Some countries specifically use the words parasol and parapluie to differentiate based on their use. There are also combinations of parasol and parapluie that are called en-tout-cas. A modern hand-held umbrella or parasol may have a black exterior canopy and a silver inner coating, for better protection from both the sun and ultraviolet rays, and may be water-resistant.
A jacket is a garment for the upper body, usually extending below the hips. A jacket typically has sleeves and fastens in the front or slightly on the side. Jackets without sleeves are vests. A jacket is generally lighter, tighter-fitting, and less insulating than a coat, but both are outerwear. Some jackets are fashionable, while some others serve as protective clothing.
A shoe is an item of footwear normally found in pairs intended to protect and comfort the human foot, usually made in such a way that one is designed to fit the left foot and the other the right foot.
A sock is a piece of clothing worn on the feet and often covering the ankle or some part of the calf. Some types of shoes or boots are typically worn over socks. In ancient times, socks were made from leather or matted animal hair. Machine-knit socks were first produced in the late 16th century. Until the 1800s, both hand-made and machine-knit socks were manufactured, with the latter technique becoming more common in the 19th century, and continuing until the modern day.
A hat is a head covering which is worn for various reasons, including protection against weather conditions, ceremonial reasons such as university graduation, religious reasons, comedy, safety, or as a fashion accessory. Hats which incorporate mechanical features, such as visors, spikes, flaps, braces or beer holders shade into the broader category of headgear.
A glove is a garment covering the hand, with separate sheaths or openings for each finger including the thumb. Gloves protect and comfort hands against cold or heat, damage by friction, abrasion or chemicals, and disease; or in turn to provide a guard for what a bare hand should not touch.
Sunglasses or sun glasses are a form of protective eyewear designed primarily to prevent bright sunlight and high-energy visible light from damaging or discomforting the eyes. They can sometimes also function as a visual aid, as variously termed spectacles or glasses exist, featuring lenses that are colored, polarized or darkened. In the early 20th century, they were also known as sun cheaters.
A watch is a timepiece carried or worn by a person. It is designed to maintain a consistent movement despite the motions caused by the person's activities. A wristwatch is worn around the wrist, attached by a watch strap or another type of bracelet, including metal bands or leather straps. A pocket watch is carried in a pocket, often attached to a chain. A stopwatch is a type of watch that measures intervals of time.
A remote control, also known colloquially as a remote or clicker, is an electronic device used to operate another device from a distance, usually wirelessly. In consumer electronics, a remote control can be used to operate devices such as a television set, DVD player or other digital home media appliance. A remote control can allow operation of devices that are out of convenient reach for direct operation of controls. They function best when used from a short distance. This is primarily a convenience feature for the user. In some cases, remote controls allow a person to operate a device that they otherwise would not be able to reach, as when a garage door opener is triggered from outside.
In electrical wiring, a light switch is a switch most commonly used to operate electric lights, permanently connected equipment, or electrical outlets. Portable lamps such as table lamps may have a light switch mounted on the socket, base, or in-line with the cord. Manually operated on/off switches may be substituted by dimmer switches that allow controlling the brightness of lamps as well as turning them on or off, time-controlled switches, occupancy-sensing switches, and remotely controlled switches and dimmers. Light switches are also found in flashlights, vehicles, and other devices.
Lamp, Lamps or LAMP may refer to:.
A pillow is a support of the body at rest for comfort, therapy, or decoration. Pillows are used in different variations by many species, including humans. Some types of pillows include throw pillows, body pillows, decorative pillows, and many more. Pillows that aid sleeping are a form of bedding that supports the head and neck. Other types of pillows are designed to support the body when lying down or sitting. There are also pillows that consider human body shape for increased comfort during sleep. Decorative pillows used on beds, couches or chairs are sometimes referred to as cushions.
A blanket is a swath of soft cloth large enough either to cover or to enfold most of the user's body and thick enough to keep the body warm by trapping radiant body heat that otherwise would be lost through convection and radiation.
Could not find summary for "Bed Sheet".
A towel is a piece of absorbent cloth, or paper, used for drying or wiping a surface. Towels draw moisture through direct contact.
A toothbrush is a special type of brush used to clean the teeth, gums, and tongue. It consists of a head of tightly clustered bristles, onto which toothpaste is applied, mounted on a handle that facilitates cleaning hard-to-reach areas of the mouth. They should be used in conjunction with tools that clean between the teeth―where toothbrush bristles cannot reach―such as floss, tape, interdental brushes or toothpicks.
Toothpaste is a paste or gel dentifrice that is used with a toothbrush to clean and maintain the aesthetics of teeth. Toothpaste is used to promote oral hygiene: it is an abrasive that aids in removing dental plaque and food from the teeth, assists in suppressing halitosis, and delivers active ingredients to help prevent tooth decay and gum disease (gingivitis). Due to variations in composition and fluoride content, not all toothpastes are equally effective in maintaining oral health. The decline of tooth decay during the 20th century has been attributed to the introduction and regular use of fluoride-containing toothpastes worldwide. Large amounts of swallowed toothpaste can be poisonous. Common colors for toothpaste include white and blue.
Soap is a salt of a fatty acid used for cleaning and lubricating products as well as other applications. In a domestic setting, soaps, specifically "toilet soaps", are surfactants usually used for washing, bathing, and other types of housekeeping. In industrial settings, soaps are used as thickeners, components of some lubricants, emulsifiers, and catalysts.
Shampoo is a hair care product, typically in the form of a viscous liquid, that is formulated to be used for cleaning (scalp) hair. Less commonly, it is available in solid bar format. Shampoo is used by applying it to wet hair, massaging the product in the hair, roots and scalp, and then rinsing it out. Some users may follow a shampooing with the use of hair conditioner.
A conditioner is something that improves the quality of another item.
A hairbrush is a brush with rigid or light and soft spokes used in hair care for smoothing, styling, and detangling human hair, or for grooming an animal's fur. It can also be used for styling in combination with a curling iron or hair dryer.
A comb is a tool consisting of a shaft that holds a row of teeth for pulling through the hair to clean, untangle, or style it. Combs have been used since prehistoric times, having been discovered in very refined forms from settlements dating back to 5,000 years ago in Persia.

A deodorant is a substance applied to the body to prevent or mask body odor caused by bacterial breakdown of perspiration, such as that in the armpits, groin, or feet. A subclass of deodorants called antiperspirants prevents sweating itself, typically by blocking sweat glands. Antiperspirants are used on a wider range of body parts at any place where sweat would be inconvenient or unsafe. Other types of deodorant allow sweating but prevent bacterial action on sweat.
A razor is a bladed tool primarily used in the removal of body hair through the act of shaving. Kinds of razors include straight razors, safety razors, disposable razors, and electric shavers.
A mirror, also known as a looking glass, is an object that reflects an image. Light that bounces off a mirror forms an image of whatever is in front of it, which is then focused through the lens of the eye or a camera. Mirrors reverse the direction of light at an angle equal to its incidence. This allows the viewer to see themselves or objects behind them, or even objects that are at an angle from them but out of their field of view, such as around a corner. Natural mirrors have existed since prehistoric times, such as the surface of water, but people have been manufacturing mirrors out of a variety of materials for thousands of years, like stone, metals, and glass. In modern mirrors, metals like silver or aluminium are often used due to their high reflectivity, applied as a thin coating on glass because of its naturally smooth and very hard surface.
A waste container, also known as a dustbin, rubbish bin, trash can, garbage can, wastepaper basket, and wastebasket, among other names, is a type of container intended to store waste. It is usually made out of metal or plastic. The words "rubbish", "basket" and "bin" are more common in British English usage; "trash" and "can" are more common in American English usage. "Garbage" may refer to food waste specifically or to municipal solid waste in general. The word "dumpster" refers to a large outdoor waste container for garbage collectors to pick up the contents.
A recycling bin is a container used to hold recyclables before they are taken to recycling centers. Recycling bins exist in various sizes for use inside and outside of homes, offices, and large public facilities. Separate containers are often provided for paper, tin or aluminum cans, and glass or plastic bottles, with some bins allowing for commingled, mixed recycling of various materials.
A broom, also known as a broomstick, is a cleaning tool, consisting of usually stiff fibers attached to, and roughly parallel to, a cylindrical handle, the broomstick. It is thus a variety of brush with a long handle. It is commonly used in combination with a dustpan.
A dustpan, the small version of which is also known as a "hearth brush and shovel”, is a cleaning utensil. The dustpan is commonly used in combination with a broom or long brush. The small dustpan may appear to be a type of flat scoop. Though often hand-held for home use, industrial and commercial enterprises use a hinged variety on the end of a long handle to allow the user to stand instead of stoop while using it.
A vacuum is space devoid of matter. The word is derived from the Latin adjective vacuus meaning "vacant" or "void". An approximation to such vacuum is a region with a gaseous pressure much less than atmospheric pressure. Physicists often discuss ideal test results that would occur in a perfect vacuum, which they sometimes simply call "vacuum" or free space, and use the term partial vacuum to refer to an actual imperfect vacuum as one might have in a laboratory or in space. In engineering and applied physics on the other hand, vacuum refers to any space in which the pressure is considerably lower than atmospheric pressure. The Latin term in vacuo is used to describe an object that is surrounded by a vacuum.
Could not find summary for "Laundry Basket".
Hanger or hangers may refer to:.
Iron is a chemical element; it has symbol Fe and atomic number 26. It is a metal that belongs to the first transition series and group 8 of the periodic table. It is, by mass, the most common element on Earth, forming much of Earth's outer and inner core. It is the fourth most abundant element in the Earth's crust. In its metallic state it was mainly deposited by meteorites.
Could not find summary for "Ironing Board".
A clock or chronometer is a device that measures and displays time. The clock is one of the oldest human inventions, meeting the need to measure intervals of time shorter than the natural units such as the day, the lunar month, and the year. Devices operating on several physical processes have been used over the millennia.
A calendar is a system of organizing days. This is done by giving names to periods of time, typically days, weeks, months and years. A date is the designation of a single and specific day within such a system. A calendar is also a physical record of such a system. A calendar can also mean a list of planned events, such as a court calendar, or a partly or fully chronological list of documents, such as a calendar of wills.
A whiteboard is a glossy, usually white surface for making non-permanent markings. Whiteboards are analogous to blackboards, but with a smoother surface allowing for rapid marking and erasing of markings on their surface. The popularity of whiteboards increased rapidly in the mid-1990s and they have become a fixture in many offices, meeting rooms, school classrooms, public events and other work environments.
The term Marker may refer to:.
Could not find summary for "Phone Case".
Could not find summary for "Screen Protector".
Could not find summary for "USB Cable".
Could not find summary for "Power Bank".
A flashlight or electric torch, usually shortened to torch, is a portable hand-held electric lamp. Formerly, the light source typically was a miniature incandescent light bulb, but these have been displaced by light-emitting diodes (LEDs) since the early 2000s. A typical flashlight consists of the light source mounted in a reflector, a transparent cover to protect the light source and reflector, a battery, and a switch, all enclosed in a case.
Battery most often refers to:Electric battery, a device that provides electrical power
Battery (crime), a crime involving unlawful physical contact.
Fan commonly refers to:Fan (machine), a machine for producing airflow, often used for cooling
Hand fan, an implement held and waved by hand to move air for cooling
Fan (person), short for fanatic; an enthusiast or supporter, especially with regard to entertainment.
Heating, ventilation, and air conditioning systems use advanced technologies to regulate temperature, humidity, and indoor air quality in residential, commercial, and industrial buildings, and in enclosed vehicles. Its goal is to provide thermal comfort and remove contaminants from the air. HVAC system design is a subdiscipline of mechanical engineering, based on the principles of thermodynamics, fluid mechanics, and heat transfer. Modern HVAC designs focus on energy efficiency and sustainability, especially with the rising demand for green building solutions. In modern construction, MEP engineers integrate HVAC systems with energy modeling techniques to optimize system performance and reduce operational costs. "Refrigeration" is sometimes added to the field's abbreviation as HVAC&R or HVACR, or "ventilation" is dropped, as in HACR.
Air conditioning, often abbreviated as A/C (US) or air con (UK), is the process of removing heat from an enclosed space to achieve a more comfortable interior temperature and, in some cases, controlling the humidity of internal air. Air conditioning can be achieved using a mechanical 'air conditioner' or through other methods, such as passive cooling and ventilative cooling. Air conditioning is a member of a family of systems and techniques that provide heating, ventilation, and air conditioning (HVAC). Heat pumps are similar in many ways to air conditioners but use a reversing valve, allowing them to both heat and cool an enclosed space.
A remote control is any device used to control a remote operation.
Router may refer to:Router (computing), a computer networking device
Router (woodworking), a rotating cutting tool
Router plane, a woodworking hand plane
Journey planner, a specialized search engine for optimal routes between locations
Michael Router, Catholic bishop in Ireland
The Routers, 1960s American instrumental group.
A modulator-demodulator, commonly referred to as a modem, is a computer hardware device that converts data from a digital format into a format suitable for an analog transmission medium such as telephone or radio. A modem transmits data by modulating one or more carrier wave signals to encode digital information, while the receiver demodulates the signal to recreate the original digital information. The goal is to produce a signal that can be transmitted easily and decoded reliably. Modems can be used with almost any means of transmitting analog signals, from LEDs to radio.
Speaker most commonly refers to:Speaker, a person who produces speech
Loudspeaker, a device that produces sound
Computer speakers.
A camera is an instrument used to capture and store images and videos, either digitally via an electronic image sensor, or chemically via a light-sensitive material such as photographic film. As a pivotal technology in the fields of photography and videography, cameras have played a significant role in the progression of visual arts, media, entertainment, surveillance, and scientific research. The invention of the camera dates back to the 19th century and has since evolved with advancements in technology, leading to a vast array of types and models in the 21st century.
A tripod is a portable three-legged frame or stand, used as a platform for supporting the weight and maintaining the stability of some other object. The three-legged design provides good stability against gravitational loads as well as horizontal shear forces, and better leverage for resisting tipping over due to lateral forces can be achieved by spreading the legs away from the vertical centre.
Variations with one, two, and four legs are termed monopod, bipod, and quadripod.
A microphone, colloquially called a mic, or mike, is a transducer that converts sound into an electrical signal. Microphones are used in telecommunication, sound recording, broadcasting, and consumer electronics, including telephones, hearing aids, and mobile devices.
Notebook Paper is the debut studio album by American rapper Huey. It was released on June 19, 2007, via Hitz Committee/Jive/Zomba Records. Production was handled by several record producers, including Jazze Pha, StarGate, T-Mix and T-Pain. It features guest appearances from Asia Cruise, Diamond, Kydd Trell, Bow Wow, Lloyd, MeMpHiTz, T-Pain, Trey Songz and Yo Gotti.
Sticky Notes is a desktop notes application included in Windows 7, Windows 8, Windows 8.1, Windows 10 and Windows 11. The app loads quickly and enables users to quickly take notes using post-it note–like windows on their desktop.
An envelope is a common packaging item, usually made of thin, flat material. It is designed to contain a flat object, such as a letter or card.
Stamp or Stamps or Stamping may refer to:.
Could not find summary for "Wallet Card".
An identity document is a document proving a person's identity.
A coin is a small object, usually round and flat, used primarily as a medium of exchange or legal tender. They are standardized in weight, and produced in large quantities at a mint in order to facilitate trade. They are most often issued by a government. Coins often have images, numerals, or text on them. The faces of coins or medals are sometimes called the obverse and the reverse, referring to the front and back sides, respectively. The obverse of a coin is commonly called heads, because it often depicts the head of a prominent person, and the reverse is known as tails.
Could not find summary for "Water Filter".
Could not find summary for "Dish Sponge".
Could not find summary for "Cutting Board".
Pan or PAN may refer to:.
Pot may refer to:.
Could not find summary for "Oven Mitt".
A measuring cup is a kitchen utensil used primarily to measure the volume of liquid or bulk solid cooking ingredients such as flour and sugar, especially for volumes from about 50 mL upwards. Measuring cups are also used to measure washing powder, liquid detergents and bleach for clothes washing. Some measuring cups will have a scale marked in cups and fractions of a cup, and often with fluid measure and weight of a selection of dry foodstuffs. Others are made to a specific capacity and are designed to be filled to the top with dry ingredients.
Could not find summary for "Measuring Spoon".
Quantum mechanics is the fundamental physical theory that describes the behavior of matter and of light; its unusual characteristics typically occur at and below the scale of atoms. It is the foundation of all quantum physics, which includes quantum chemistry, quantum biology, quantum field theory, quantum technology, and quantum information science.
General relativity, also known as the general theory of relativity, and as Einstein's theory of gravity, is the geometric theory of gravitation published by Albert Einstein in May 1916 and is the accepted description of the gravitation of macroscopic objects in modern physics. General relativity generalizes special relativity and refines Isaac Newton's law of universal gravitation, providing a unified description of gravity as a geometric property of space and time, or four-dimensional spacetime. In particular, the curvature of spacetime is directly related to the energy, momentum, and stress of whatever is present, including matter and radiation. The relation is specified by the Einstein field equations, a system of second-order partial differential equations. John Archibald Wheeler summarized it: "Space-time tells matter how to move; matter tells space-time how to curve.".
In physics, the special theory of relativity, or special relativity for short, is a scientific theory of the relationship between space and time. In Albert Einstein's 1905 paper,
"On the Electrodynamics of Moving Bodies", the theory is presented as being based on just two postulates:The laws of physics are invariant (identical) in all inertial frames of reference. This is known as the principle of relativity.
The speed of light in vacuum is the same for all observers, regardless of the motion of light source or observer. This is known as the principle of light constancy, or the principle of light speed invariance.
In physics, classical mechanics is a theory that describes the effect of forces on the motion of macroscopic objects and bulk matter, without considering quantum effects, and often without incorporating relativistic effects either.
Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation. The behavior of these quantities is governed by the four laws of thermodynamics, which convey a quantitative description using measurable macroscopic physical quantities but may be explained in terms of microscopic constituents by statistical mechanics. Thermodynamics applies to various topics in science and engineering, especially physical chemistry, biochemistry, chemical engineering, and mechanical engineering, as well as other complex fields such as meteorology.
In physics, statistical mechanics is a mathematical framework that applies statistical methods and probability theory to large assemblies of microscopic entities. Sometimes called statistical physics or statistical thermodynamics, its applications include many problems in a wide variety of fields such as biology, neuroscience, computer science, information theory and sociology. Its main purpose is to clarify the properties of matter in aggregate, in terms of physical laws governing atomic motion.
In physics, electromagnetism is an interaction that occurs between particles with electric charge via electromagnetic fields. The electromagnetic force is one of the four fundamental forces of nature. It is the dominant force in the interactions of atoms and molecules. Electromagnetism can be thought of as a combination of electrostatics and magnetism, which are distinct but closely intertwined phenomena. Electromagnetic forces occur between any two charged particles. Electric forces cause an attraction between particles with opposite charges and repulsion between particles with the same charge, while magnetism is an interaction that occurs between charged particles in relative motion. These two forces are described in terms of electromagnetic fields. Macroscopic charged objects are described in terms of Coulomb's law for electricity and Ampère's force law for magnetism; the Lorentz force describes microscopic charged particles.
In theoretical physics, quantum field theory (QFT) is a theoretical framework that combines field theory, special relativity and quantum mechanics. QFT is used in particle physics to construct physical models of subatomic particles and in condensed matter physics to construct models of quasiparticles. The current standard model of particle physics is based on QFT.
Particle physics or high-energy physics is the study of fundamental particles and forces that constitute matter and radiation. The field also studies combinations of elementary particles up to the scale of protons and neutrons, while the study of combinations of protons and neutrons is called nuclear physics.
Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions, in addition to the study of other forms of nuclear matter.
Astrophysics is a science that applies the methods and principles of physics and chemistry in the study of astronomical objects and phenomena including the universe. As one of the founders of the discipline, James Keeler, said, astrophysics "seeks to ascertain the nature of the heavenly bodies, rather than their positions or motions in space—what they are, rather than where they are", which is studied in celestial mechanics.
Cosmology is the study of the nature of the universe, the cosmos. The term cosmology was first used in English in 1656 in Thomas Blount's Glossographia, with the meaning of "a speaking of the world". In 1731, German philosopher Christian Wolff used the term cosmology in Latin (cosmologia) to denote a branch of metaphysics that deals with the general nature of the physical world. Cosmology is investigated by scientists, including astronomers and physicists, as well as philosophers, such as metaphysicians, philosophers of physics, and philosophers of space and time. Because of this shared scope with philosophy, theories in physical cosmology may include both scientific and non-scientific propositions and may depend upon assumptions that cannot be tested. Religious or mythological cosmology is a body of beliefs based on mythological, religious, and esoteric literature and traditions of creation myths and eschatology.
Stellar evolution is the process by which a star changes over the course of time. Depending on the mass of the star, its lifetime can range from a few million years for the most massive to trillions of years for the least massive, which is considerably longer than the current age of the universe. The table shows the lifetimes of stars as a function of their masses. All stars are formed from collapsing clouds of gas and dust, often called nebulae or molecular clouds. Over the course of millions of years, these protostars settle down into a state of equilibrium, becoming what is known as a main sequence star.
Planetary science is the scientific study of planets, celestial bodies and planetary systems and the processes of their formation. It studies objects ranging in sizes from micrometeoroids to huge gas giants, with the aim of determining their composition, dynamics, formation, interrelations and history. It is a strongly interdisciplinary field, which originally grew from astronomy and Earth science, and now incorporates many disciplines, including planetary geology, cosmochemistry, atmospheric science, physics, oceanography, hydrology, theoretical planetary science, glaciology, and exoplanetology. Allied disciplines include space physics, when concerned with the effects of the Sun on the bodies of the Solar System, and astrobiology.
Could not find summary for "Exoplanet Research".
Could not find summary for "Galactic Dynamics".
Could not find summary for "Black Hole Physics".
In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings. String theory describes how these strings propagate through space and interact with each other. On distance scales larger than the string scale, a string acts like a particle, with its mass, charge, and other properties determined by the vibrational state of the string. In string theory, one of the many vibrational states of the string corresponds to the graviton, a quantum mechanical particle that carries the gravitational force. Thus, string theory is a theory of quantum gravity.
Chaos theory is an interdisciplinary area of scientific study and branch of mathematics. It focuses on underlying patterns and deterministic laws of dynamical systems that are highly sensitive to initial conditions. These were once thought to have completely random states of disorder and irregularities. Chaos theory states that within the apparent randomness of chaotic complex systems, there are underlying patterns, interconnection, constant feedback loops, repetition, self-similarity, fractals and self-organization. The butterfly effect, an underlying principle of chaos, describes how a small change in one state of a deterministic nonlinear system can result in large differences in a later state. A metaphor for this behavior is that a butterfly flapping its wings in Brazil can cause or prevent a tornado in Texas.
A complex system is a system composed of many components that interact with one another. Examples of complex systems are Earth's global climate, organisms, the human brain, infrastructure such as power grid, transportation or communication systems, complex software and electronic systems, social and economic organizations, an ecosystem, a living cell, and, ultimately, for some authors, the entire universe.
Evolutionary biology is a subfield of biology that analyzes the four mechanisms of evolution: natural selection, mutation, genetic drift, and gene flow. The purpose of evolutionary biology is to observe the diversity of life on Earth. The idea of natural selection was first researched by Charles Darwin as he studied bird beaks. The discipline of evolutionary biology emerged through what Julian Huxley called the modern synthesis of understanding, from previously unrelated fields of biological research, such as genetics and ecology, systematics, and paleontology. Huxley was able to take what Charles Darwin discovered and elaborate to build on his understandings.
Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene.
Molecular biology is a branch of biology that seeks to understand the molecular structures and chemical processes that are the basis of biological activity within and between cells. It is centered largely on the study of nucleic acids and proteins. It examines the structure, function, and interactions of these macromolecules as they orchestrate processes such as replication, transcription, translation, protein synthesis, and complex biomolecular interactions. The field of molecular biology is multi-disciplinary, relying on principles from genetics, biochemistry, physics, mathematics, and more recently computer science (bioinformatics).
Cell biology, cellular biology, or cytology, is the branch of biology that studies the structure, function, and behavior of the cells. All organisms are made of cells. A cell is the basic unit of life that is responsible for the living and functioning of an organism. Cell biology encompasses both prokaryotic and eukaryotic cells, with subtopics including the study of cell metabolism, cell communication, cell cycle, biochemistry, and cell composition.
Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences.

Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.".
Biochemistry, or biological chemistry, is the study of chemical processes within and relating to living organisms. A sub-discipline of both chemistry and biology, biochemistry may be divided into three fields: structural biology, enzymology, and metabolism. Over the last decades of the 20th century, biochemistry has become successful at explaining living processes through these three disciplines. Almost all areas of the life sciences are being uncovered and developed through biochemical methodology and research. Biochemistry focuses on understanding the chemical basis that allows biological molecules to give rise to the processes that occur within living cells and between cells, in turn relating greatly to the understanding of tissues and organs as well as organism structure and function. Biochemistry is closely related to molecular biology, the study of the molecular mechanisms of biological phenomena.
Biophysics is an interdisciplinary science that applies approaches and methods traditionally used in physics to study biological phenomena.
Microbiology is the scientific study of microorganisms, those being of unicellular (single-celled), multicellular, or acellular. Microbiology encompasses numerous sub-disciplines including virology, bacteriology, protistology, mycology, immunology, and parasitology.
Virology is the scientific study of biological viruses. It is a subfield of microbiology that focuses on their detection, structure, classification and evolution, their methods of infection and exploitation of host cells for reproduction, their interaction with host organism physiology and immunity, the diseases they cause, the techniques to isolate and culture them, and their use in research and therapy.
Immunology is a branch of biology and medicine that covers the study of immune systems in all organisms.
Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history.
Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology.
Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry.
Oceanography, also known as oceanology, sea science, ocean science, and marine science, is the scientific study of the ocean, including its physics, chemistry, biology, and geology.
Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science.
Volcanology is the study of volcanoes, lava, magma and related geological, geophysical and geochemical phenomena (volcanism). The term volcanology is derived from the Latin word vulcan. Vulcan was the ancient Roman god of fire.
Seismology is the scientific study of earthquakes and the generation and propagation of elastic waves through planetary bodies. It also includes studies of the environmental effects of earthquakes such as tsunamis; other seismic sources such as volcanoes, plate tectonics, glaciers, rivers, oceanic microseisms, and the atmosphere; and artificial processes such as explosions.
Paleontology or palaeontology is the scientific study of the life of the past, mainly but not exclusively through the study of fossils. Paleontologists use fossils as a means to classify organisms, measure geologic time, and assess the interactions between prehistoric organisms and their natural environment. While paleontological observations are known from at least the 6th century BC, the foundation of paleontology as a science dates back to the work of Georges Cuvier in 1796. Cuvier demonstrated evidence for the concept of extinction and how the life of the past was not necessarily the same as that of the present. The field developed rapidly over the course of the following decades, and the French word paléontologie was introduced for the study in 1822, which was derived from the Ancient Greek word for 'ancient' and words describing relatedness and a field of study. Further advances in the field accompanied the work of Charles Darwin who popularized the concept of evolution. Together, evolution and extinction can be understood as complementary processes that shaped the history of life.
Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research.
Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives.
Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
Polymer science or macromolecular science is a subfield of materials science concerned with polymers, primarily synthetic polymers such as plastics and elastomers. The field of polymer science includes researchers in multiple disciplines including chemistry, physics, and engineering.
Crystallography is the branch of science devoted to the study of molecular and crystalline structure and properties. The word crystallography is derived from the Ancient Greek word κρύσταλλος, and γράφειν. In July 2012, the United Nations recognised the importance of the science of crystallography by proclaiming 2014 the International Year of Crystallography.
Organic chemistry is a subdiscipline within chemistry involving the scientific study of the structure, properties, and reactions of organic compounds and organic materials. It involves studying the structure of organic material to determine the structural formula, analyzing physical and chemical properties, and evaluating chemical reactivity to understand the behavior of organic compounds. The study of organic reactions includes the chemical synthesis of natural products, drugs, and polymers, and study of individual organic molecules in the laboratory and via theoretical study.
Inorganic chemistry deals with synthesis and behavior of inorganic and organometallic compounds. This field covers chemical compounds that are not carbon-based, which are the subjects of organic chemistry. The distinction between the two disciplines is far from absolute, as there is much overlap in the subdiscipline of organometallic chemistry. It has applications in every aspect of the chemical industry, including catalysis, materials science, pigments, surfactants, coatings, medications, fuels, and agriculture.
Physical chemistry is the study of macroscopic and microscopic phenomena in chemical systems in terms of the principles, practices, and concepts of physics such as motion, energy, force, time, thermodynamics, quantum chemistry, statistical mechanics, analytical dynamics and chemical equilibria.
Analytical chemistry is the branch of chemistry concerned with the development and application of methods to identify the chemical composition of materials and quantify the amounts of components in mixtures. It focuses on methods to identify unknown compounds, possibly in a mixture or solution, and quantify a compound's presence in terms of amount of substance, concentration, percentage by mass or number of moles in a mixture of compounds.
Computational chemistry is a branch of chemistry that uses computer simulations to assist in solving chemical problems. It uses methods of theoretical chemistry incorporated into computer programs to calculate the structures and properties of molecules, groups of molecules, and solids. The importance of this subject stems from the fact that, with the exception of some relatively recent findings related to the hydrogen molecular ion, achieving an accurate quantum mechanical depiction of chemical systems analytically, or in a closed form, is not feasible. The complexity inherent in the many-body problem exacerbates the challenge of providing detailed descriptions of quantum mechanical systems. While computational results normally complement information obtained by chemical experiments, it can occasionally predict unobserved chemical phenomena.
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
In machine learning, deep learning (DL) focuses on utilizing multilayered neural networks to perform tasks such as classification, regression, and representation learning. The field takes inspiration from biological neuroscience and revolves around stacking artificial neurons into layers and "training" them to process data. The adjective "deep" refers to the use of multiple layers in the network. Methods used can be supervised, semi-supervised or unsupervised.
Computer vision tasks include methods for acquiring, processing, analyzing, and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, e.g. in the form of decisions. "Understanding" in this context signifies the transformation of visual images into descriptions of the world that make sense to thought processes and can elicit appropriate action. This image understanding can be seen as the disentangling of symbolic information from image data using models constructed with the aid of geometry, physics, statistics, and learning theory.
Natural language processing (NLP) is the processing of natural language information by a computer. NLP is a subfield of computer science and is closely associated with artificial intelligence. NLP is also related to information retrieval, knowledge representation, computational linguistics, and linguistics more broadly.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. Robotics usually combines four aspects of design work: a power source, mechanical construction, a control system, and software.
Cybernetics is the transdisciplinary study of circular causal processes such as feedback and recursion, where the effects of a system's actions return as inputs to that system, influencing subsequent actions. It is concerned with general principles that are relevant across multiple contexts, including engineering, ecological, economic, biological, cognitive and social systems and also in practical activities such as designing, learning, and managing. Cybernetics' transdisciplinary character means that it intersects with a number of other fields, resulting in a wide influence and diverse interpretations.
Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering.
Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications.
A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems. These systems evolve in ways that operate on an enormous number of possibilities simultaneously, though they remain subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks.
Bioinformatics is an interdisciplinary field of science that develops computational methods and software tools for understanding biological data, especially when the data sets are large and complex. Bioinformatics uses biology, chemistry, physics, computer science, data science, computer programming, information engineering, mathematics and statistics to analyze and interpret biological data. This process can sometimes be referred to as computational biology, however the distinction between the two terms is often disputed. To some, the term computational biology refers to building and using models of biological systems.
Systems biology is the computational and mathematical analysis and modeling of complex biological systems. It is a biology-based interdisciplinary field of study that focuses on complex interactions within biological systems, using a holistic approach to biological research. This multifaceted research domain necessitates the collaborative efforts of chemists, biologists, mathematicians, physicists, and engineers to decipher the biology of intricate living systems by merging various quantitative molecular measurements with carefully constructed mathematical models. It represents a comprehensive method for comprehending the complex relationships within biological systems. In contrast to conventional biological studies that typically center on isolated elements, systems biology seeks to combine different biological data to create models that illustrate and elucidate the dynamic interactions within a system. This methodology is essential for understanding the complex networks of genes, proteins, and metabolites that influence cellular activities and the traits of organisms. One of the aims of systems biology is to model and discover emergent properties, of cells, tissues and organisms functioning as a system whose theoretical description is only possible using techniques of systems biology. By exploring how function emerges from dynamic interactions, systems biology bridges the gaps that exist between molecules and physiological processes.
Synthetic biology (SynBio) is a multidisciplinary field of science that focuses on living systems and organisms. It applies engineering principles to develop new biological parts, devices, and systems or to redesign existing systems found in nature.
Genetic engineering, also called genetic modification or genetic manipulation, is the modification and manipulation of an organism's genes using technology. It is a set of technologies used to change the genetic makeup of cells, including the transfer of genes within and across species boundaries to produce improved or novel organisms. New DNA is obtained by either isolating and copying the genetic material of interest using recombinant DNA methods or by artificially synthesising the DNA. A construct is usually created and used to insert this DNA into the host organism. The first recombinant DNA molecule was designed by Paul Berg in 1972 by combining DNA from the monkey virus SV40 with the lambda virus. As well as inserting genes, the process can be used to remove, or "knock out", genes. The new DNA can either be inserted randomly or targeted to a specific part of the genome.
Could not find summary for "CRISPR Technology".
Pharmacology is the science of drugs and medications, including a substance's origin, composition, pharmacokinetics, pharmacodynamics, therapeutic use, and toxicology. More specifically, it is the study of the interactions that occur between a living organism and chemicals that affect normal or abnormal biochemical function. If substances have medicinal properties, they are considered pharmaceuticals.
Toxicology is a scientific discipline, overlapping with biology, chemistry, pharmacology, and medicine, that involves the study of the adverse effects of chemical substances on living organisms and the practice of diagnosing and treating exposures to toxins and toxicants. The relationship between dose and its effects on the exposed organism is of high significance in toxicology. Factors that influence chemical toxicity include the dosage, duration of exposure, route of exposure, species, age, sex, and environment. Toxicologists are experts on poisons and poisoning. There is a movement for evidence-based toxicology as part of the larger movement towards evidence-based practices. Toxicology is currently contributing to the field of cancer research, since some toxins can be used as drugs for killing tumor cells. One prime example of this is ribosome-inactivating proteins, tested in the treatment of leukemia.
Neuropharmacology is the study of how drugs affect function in the nervous system, and the neural mechanisms through which they influence behavior. There are two main branches of neuropharmacology: behavioral and molecular. Behavioral neuropharmacology focuses on the study of how drugs affect human behavior (neuropsychopharmacology), including the study of how drug dependence and addiction affect the human brain. Molecular neuropharmacology involves the study of neurons and their neurochemical interactions, with the overall goal of developing drugs that have beneficial effects on neurological function. Both of these fields are closely connected, since both are concerned with the interactions of neurotransmitters, neuropeptides, neurohormones, neuromodulators, enzymes, second messengers, co-transporters, ion channels, and receptor proteins in the central and peripheral nervous systems. Studying these interactions, researchers are developing drugs to treat many different neurological disorders, including pain, neurodegenerative diseases such as Parkinson's disease and Alzheimer's disease, psychological disorders, addiction, and many others.
Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole.
Radio astronomy is a subfield of astronomy that studies celestial objects using radio waves. It started in 1933, when Karl Jansky at Bell Telephone Laboratories reported radiation coming from the Milky Way. Subsequent observations have identified a number of different sources of radio emission. These include stars and galaxies, as well as entirely new classes of objects, such as radio galaxies, quasars, pulsars, and masers. The discovery of the cosmic microwave background radiation, regarded as evidence for the Big Bang theory, was made through radio astronomy.
Optics is the branch of physics that studies the behaviour, manipulation, and detection of electromagnetic radiation, including its interactions with matter and instruments that use or detect it. Optics usually describes the behaviour of visible, ultraviolet, and infrared light. The study of optics extends to other forms of electromagnetic radiation, including radio waves, microwaves,
and X-rays. The term optics is also applied to technology for manipulating beams of elementary charged particles.
Photonics is a branch of optics that involves the application of generation, detection, and manipulation of light in the form of photons through emission, transmission, modulation, signal processing, switching, amplification, and sensing. Even though photonics is a commonly used term, there is no widespread agreement on a clear definition of the term or on the difference between photonics and related fields, such as optics.
Acoustics is a branch of continuum mechanics that deals with the study of mechanical waves in gases, liquids, and solids including topics such as vibration, sound, ultrasound and infrasound. A scientist who works in the field of acoustics is an acoustician while someone working in the field of acoustics technology may be called an acoustical engineer. The application of acoustics is present in almost all aspects of modern society with the most obvious being the audio and noise control industries.
In physics, physical chemistry, and engineering, fluid dynamics is a subdiscipline of fluid mechanics that describes the flow of fluids – liquids and gases. It has several subdisciplines, including aerodynamics and hydrodynamics. Fluid dynamics has a wide range of applications, including calculating forces and moments on aircraft, determining the mass flow rate of petroleum through pipelines, predicting weather patterns, understanding nebulae in interstellar space, understanding large scale geophysical flows involving oceans/atmosphere and modelling fission weapon detonation.
Aerodynamics is the study of the motion of air, particularly when affected by a solid object, such as an airplane wing. It involves topics covered in the field of fluid dynamics and its subfield of gas dynamics, and is an important domain of study in aeronautics. The term aerodynamics is often used synonymously with gas dynamics, the difference being that "gas dynamics" applies to the study of the motion of all gases, and is not limited to air. The formal study of aerodynamics began in the modern sense in the eighteenth century, although observations of fundamental concepts such as aerodynamic drag were recorded much earlier. Most of the early efforts in aerodynamics were directed toward achieving heavier-than-air flight, which was first demonstrated by Otto Lilienthal in 1891. Since then, the use of aerodynamics through mathematical analysis, empirical approximations, wind tunnel experimentation, and computer simulations has formed a rational basis for the development of heavier-than-air flight and a number of other technologies. Recent work in aerodynamics has focused on issues related to compressible flow, turbulence, and boundary layers and has become increasingly computational in nature.
Plasma is a state of matter that results from a gaseous state having undergone some degree of ionization. It thus consists of a significant portion of charged particles. While rarely encountered on Earth, it is estimated that 99.9% of all ordinary matter in the universe is plasma. Stars are almost pure balls of plasma, and plasma dominates the rarefied intracluster medium and intergalactic medium. Plasma can be artificially generated, for example, by heating a neutral gas or subjecting it to a strong electromagnetic field.
Could not find summary for "Energy Science".
Renewable energy is energy made from renewable natural resources that are replenished on a human timescale. The most widely used renewable energy types are solar energy, wind power, and hydropower. Bioenergy and geothermal power are also significant in some countries. Renewable energy installations can be large or small and are suited for both urban and rural areas. Renewable energy is often deployed together with further electrification. This has several benefits: electricity can move heat and vehicles efficiently and is clean at the point of consumption. Variable renewable energy sources are those that have a fluctuating nature, such as wind power and solar power. In contrast, controllable renewable energy sources include dammed hydroelectricity, bioenergy, or geothermal power.
Nuclear fusion is a reaction in which two or more atomic nuclei combine to form a larger nucleus. The difference in mass between the reactants and products is manifested as either the release or the absorption of energy. This difference in mass arises as a result of the difference in nuclear binding energy between the atomic nuclei before and after the fusion reaction. Nuclear fusion is the process that powers all active stars, via many reaction pathways.
Could not find summary for "Space Engineering".
Aerospace engineering is the primary field of engineering concerned with the development of aircraft and spacecraft. It has two major and overlapping branches: aeronautical engineering and astronautical engineering. Avionics engineering is similar, but deals with the electronics side of aerospace engineering.
The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global.
Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use.
Chemical engineering is an engineering field which deals with the study of the operation and design of chemical plants as well as methods of improving production. Chemical engineers develop economical commercial processes to convert raw materials into useful products. Chemical engineering uses principles of chemistry, physics, mathematics, biology, and economics to efficiently use, produce, design, transport and transform energy and materials. The work of chemical engineers can range from the utilization of nanotechnology and nanomaterials in the laboratory to large-scale industrial processes that convert chemicals, raw materials, living cells, microorganisms, and energy into useful forms and products. Chemical engineers are involved in many aspects of plant design and operation, including safety and hazard assessments, process design and analysis, modeling, control engineering, chemical reaction engineering, nuclear engineering, biological engineering, construction specification, and operating instructions.
Biomedical engineering (BME) or medical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare applications. BME also integrates the logical sciences to advance health care treatment, including diagnosis, monitoring, and therapy. Also included under the scope of a biomedical engineer is the management of current medical equipment in hospitals while adhering to relevant industry standards. This involves procurement, routine testing, preventive maintenance, and making equipment recommendations, a role also known as a Biomedical Equipment Technician (BMET) or as a clinical engineer.
Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways.
Structural engineering is a sub-discipline of civil engineering in which structural engineers are trained to design the 'bones and joints' that create the form and shape of human-made structures. Structural engineers also must understand and calculate the stability, strength, rigidity and earthquake-susceptibility of built structures for buildings and nonbuilding structures. The structural designs are integrated with those of other designers such as architects and building services engineer and often supervise the construction of projects by contractors on site. They can also be involved in the design of machinery, medical equipment, and vehicles where structural integrity affects functioning and safety. See glossary of structural engineering.

A mathematical model is an abstract description of a concrete system using mathematical concepts and language. The process of developing a mathematical model is termed mathematical modeling. Mathematical models are used in many fields, including applied mathematics, natural sciences, social sciences and engineering. In particular, the field of operations research studies the use of mathematical modelling and related tools to solve problems in business or military operations. A model may help to characterize a system by studying the effects of different components, which may be used to make predictions about behavior or solve specific problems.
Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
Number theory is a branch of pure mathematics devoted primarily to the study of the integers and arithmetic functions. Number theorists study prime numbers as well as the properties of mathematical objects constructed from integers, or defined as generalizations of the integers.
Probability theory or probability calculus is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms. Typically these axioms formalise probability in terms of a probability space, which assigns a measure taking values between 0 and 1, termed the probability measure, to a set of outcomes called the sample space. Any specified subset of the sample space is called an event.
Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers.
Econometrics is an application of statistical methods to economic data in order to give empirical content to economic relationships. More precisely, it is "the quantitative analysis of actual economic phenomena based on the concurrent development of theory and observation, related by appropriate methods of inference." An introductory economics textbook describes econometrics as allowing economists "to sift through mountains of data to extract simple relationships." Jan Tinbergen is one of the two founding fathers of econometrics. The other, Ragnar Frisch, also coined the term in the sense in which it is used today.
Social physics or sociophysics is an interdisciplinary field of science which uses mathematical tools inspired by physics to understand the behavior of human crowds. In a modern commercial use, it can also refer to the analysis of social phenomena with big data.

Behavioural science is the branch of science concerned with theorizing on, categorizing, and judging human behaviour. It sits in the interstice between fields such as psychology, cognitive science, neuroscience, behavioral biology, behavioral genetics and social science. While the term can technically be applied to the study of behaviour amongst all living organisms, it is nearly always used with reference to humans as the primary target of investigation.
Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions.
Could not find summary for "Cognitive Robotics".
Astrobiology is a scientific field within the life and environmental sciences that studies the origins, early evolution, distribution, and future of life in the universe by investigating its deterministic conditions and contingent events. As a discipline, astrobiology is founded on the premise that life may exist beyond Earth.
Could not find summary for "Exochemistry".
Egypt, officially the Arab Republic of Egypt, is a country spanning the northeast corner of Africa and southwest corner of Asia via the Sinai Peninsula. It is bordered by the Mediterranean Sea to the north, Palestine and Israel to the northeast, the Red Sea to the east, Sudan and the Sahara to the south, and Libya to the west. The Gulf of Aqaba in the northeast separates Egypt from Jordan and Saudi Arabia. Cairo is the capital, largest city, and leading cultural centre, while Alexandria is the second-largest city and an important hub of industry and tourism. With over 107 million inhabitants, Egypt is the most populous country in the Arab world, third-most populous country in Africa, and 15th-most populated in the world.
Mesopotamia is a historical region of West Asia situated within the Tigris–Euphrates river system, in the northern part of the Fertile Crescent. It corresponds roughly to the territory of modern Iraq. Just beyond it lies southwestern Iran, where the region transitions into the Persian plateau, marking the shift from the Arab world to Iran.
Iran, officially the Islamic Republic of Iran, and also known as Persia, is a country in West Asia. It borders Iraq to the west, Turkey, Azerbaijan, and Armenia to the northwest, the Caspian Sea to the north, Turkmenistan to the northeast, Afghanistan to the east, Pakistan to the southeast, and the Gulf of Oman and the Persian Gulf to the south. With a population of over 90 million, Iran ranks 17th globally in both geographic size and population and is the sixth-largest country in Asia. It is divided into five regions with 31 provinces. Tehran is the nation's capital, largest city, and financial center.
Greece, officially the Hellenic Republic, is a country in Southeast Europe. Located on the southern tip of the Balkan peninsula, it shares land borders with Albania to the northwest, North Macedonia and Bulgaria to the north, and Turkey to the east. The Aegean Sea lies to the east of the mainland, the Ionian Sea to the west, and the Sea of Crete and the Mediterranean Sea to the south. Greece has the longest coastline on the Mediterranean basin, spanning thousands of islands and nine traditional geographic regions. It has a population of over 10 million. Athens is the nation's capital and largest city, followed by Thessaloniki and Patras.
Rome is the capital city and most populated comune (municipality) of Italy. It is also the administrative centre of the Lazio region and of the Metropolitan City of Rome. A special comune named Roma Capitale with a population of 2.7 million in an area of 1,287.36 km2 (497.1 mi2), Rome is the third most populous city in the European Union by population within city limits. The Metropolitan City of Rome Capital, with a population of 4.2 million, is the most populous metropolitan city in Italy. Its metropolitan area is the third-most populous within Italy. Rome is located in the central-western portion of the Italian Peninsula, within Lazio (Latium), along the shores of the Tiber Valley. Vatican City is an independent country inside the city boundaries of Rome, the only existing example of a country within a city. Rome is often referred to as the "City of Seven Hills" due to its geography, and also as the "Eternal City". Rome is generally considered to be one of the cradles of Western civilization and Western Christian culture, and the centre of the Catholic Church.
Byzantium or Byzantion was an ancient Greek city in classical antiquity that became known as Constantinople in late antiquity and Istanbul in modern times. The Greek name Byzantion and its Latinization Byzantium continued to be used as a name of Constantinople sporadically and to varying degrees during the thousand-year existence of the Eastern Roman Empire, which also became known by the former name of the city as the Byzantine Empire. Byzantium was colonized by Greeks from Megara in the 7th century BCE and remained primarily Greek-speaking until its conquest by the Ottoman Empire in 1453 CE.
Ottoman may refer to:Osman I, historically known in English as "Ottoman I", founder of the Ottoman Empire
Osman II, historically known in English as "Ottoman II"
Osman III, historically known in English as "Ottoman III"
Ottoman Empire 1299–1922
Ottoman dynasty, ruling family of the Ottoman Empire
Osmanoğlu family, modern members of the family
Ottoman Caliphate 1517–1924
Ottoman Turks, a Turkic ethnic group
Ottoman architecture
Ottoman bed, a type of storage bed
Ottoman (furniture), padded stool or footstool
Ottoman (textile), fabric with a pronounced ribbed or corded effect, often made of silk or a mixture.
Mongols are an East Asian ethnic group native to Mongolia and China, as well as the republics of Buryatia and Kalmykia in Russia. The Mongols are the principal member of the large family of Mongolic peoples. The Oirats and the Buryats are classified either as distinct ethno-linguistic groups or as subgroups of Mongols.
China, officially the People's Republic of China (PRC), is a country in East Asia. It is the second-most populous country after India, with a population exceeding 1.4 billion, representing 17% of the world's population. China borders fourteen countries by land across an area of 9.6 million square kilometers (3,700,000 sq mi), making it the third-largest country by area. The country is divided into 33 province-level divisions: 22 provinces, 5 autonomous regions, 4 municipalities, and 2 semi-autonomous special administrative regions. Beijing is the capital, while Shanghai is the most populous city by urban area and largest financial center.
Japan is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered to the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. The Japanese archipelago consists of four major islands alongside 14,121 smaller islands. Japan is divided into 47 administrative prefectures and eight traditional regions, and around 75% of its terrain is mountainous and heavily forested, concentrating its agriculture and highly urbanized population along its eastern coastal plains. With a population of almost 123 million as of 2026, it is the world's 11th most populous country. Tokyo is the country's capital and largest city.
Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands. Since the end of World War II in Asia in 1945, it has been politically divided at or near the 38th parallel between North Korea and South Korea. Both countries proclaimed independence in 1948, and the two countries fought the Korean War from 1950 to 1953. The region is bordered by China to the north and Russia to the northeast, across the Amnok (Yalu) and Duman (Tumen) rivers, and is separated from Japan to the southeast by the Korea Strait.
India, officially the Republic of India, is a country in South Asia. It is the seventh-largest country by area; the most populous country since 2023; and, since its independence in 1947, the world's most populous democracy. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand, and Indonesia.
Maya may refer to:.
The Aztecs were a Mesoamerican civilization that flourished in central Mexico from 1300 to 1521. The Aztec people included different ethnic groups of central Mexico, particularly those groups who spoke the Nahuatl language. Aztec culture was organized into city-states (altepetl), some of which joined to form alliances, political confederations, or empires. The Aztec Empire was a confederation of three city-states established in 1427: Tenochtitlan, Tetzcoco, and Tlacopan, previously part of the Tepanec empire, whose dominant power was Azcapotzalco. Although the term Aztecs is often narrowly restricted to the Mexica of Tenochtitlan, it is also broadly used to refer to Nahua polities or peoples of central Mexico in the prehispanic era, as well as the Spanish colonial era (1521–1821).
The Inca Empire, officially known as the Realm of the Four Parts, was the largest empire in pre-Columbian America. The administrative, political, and military center of the empire was in the city of Cusco. The Inca civilisation rose from the Peruvian highlands sometime in the early 13th century. The Portuguese explorer Aleixo Garcia was the first European to reach the Inca Empire in 1524. Later, in 1532, the Spanish began the conquest of the Inca Empire, and by 1572 the last Inca state was fully conquered.
Vikings were a seafaring people originally from Scandinavia, who from the late 8th to the late 11th centuries raided, pirated, traded, and settled throughout parts of Europe. They voyaged as far as the Mediterranean, North Africa, the Middle East, Greenland, and Vinland. In their countries of origin, and in some of the countries they raided and settled, this period of activity is popularly known as the Viking Age, and the term "Viking" also commonly includes the inhabitants of the Scandinavian homelands as a whole during the late 8th to the mid-11th centuries. The Vikings had a profound impact on the early medieval history of northern and Eastern Europe, including the political and social development of England and parts of France, and the establishment of Kievan Rus', the ancestor of the later states of Belarus, Russia, and Ukraine.
The Crusades were a series of military campaigns launched by the papacy between 1095 and 1291 against Muslim rulers for the recovery and defence of the Holy Land, encouraged by promises of spiritual reward. The First Crusade was proclaimed by Pope Urban II at the Council of Clermont in November 1095—a call to arms for Christians to reconquer Jerusalem from the Muslims. By this time, the papacy's position as head of the Catholic Church had strengthened, and earlier conflicts with secular rulers and wars on the frontiers of Western Christendom had prepared it for the direction of armed force in religious causes. The successes of the First Crusade led to the establishment of four Crusader states in the Levant, where their defence required further expeditions from Catholic Europe. The organisation of such large-scale campaigns demanded complex religious, social, and economic institutions, including crusade indulgences, military orders, and the taxation of clerical income. Over time, the crusading movement expanded to include campaigns against pagans, Christian dissidents, and other enemies of the papacy, promoted with similar spiritual rewards and continuing into the 18th century.
The Renaissance is a European period of history and cultural movement, very roughly defined as covering the 14th through 17th centuries, though sometimes more narrowly defined for instance as only covering the 15th through 16th centuries. It marked the transition from the Middle Ages to modernity and was characterized by the European rediscovery and revival of the literary, philosophical, and artistic achievements of classical antiquity. Associated with great social change in most fields and disciplines, including art, architecture, politics, literature, exploration and science, the Renaissance was first centered in the Republic of Florence, then spread to the rest of Italy and later throughout Europe. The term rinascita ('rebirth') first appeared in Lives of the Artists by Giorgio Vasari, while the corresponding French word renaissance was adopted into English as the term for this period during the 1830s.
The Reformation, also known as the Protestant Reformation or the European Reformation, was a time of major theological movement in Western Christianity in 16th-century Europe that posed a religious and political challenge to the papacy and the authority of the Catholic Church hierarchy. Towards the end of the Renaissance, the Reformation marked the beginning of Protestantism. It is considered one of the events that signified the end of the Middle Ages and the beginning of the early modern period in Europe.
Enlightenment or enlighten may refer to:.
Colonialism is the practice of extending and maintaining political, social, economic, and cultural domination over a territory and its people by another people in pursuit of interests defined in an often distant metropole, who also claim superiority. While frequently an imperialist project, colonialism functions through differentiating between the targeted land and people, and that of the colonizers. Rather than annexation, this typically culminates in organizing the colonized into colonies separate to the colonizers' metropole. Colonialism sometimes deepens by developing settler colonialism, whereby settlers from one or multiple colonizing metropoles occupy a territory with the intention of partially or completely supplanting the existing indigenous peoples, possibly amounting to genocide.
Imperialism is the maintaining and extending of power over foreign nations, particularly through expansionism, employing both hard power and soft power. Imperialism focuses on establishing or maintaining hegemony and a more formal empire.
In political science, a revolution is a rapid, fundamental transformation of a society's class, state, ethnic or religious structures. According to sociologist Jack Goldstone, all revolutions contain "a common set of elements at their core: (a) efforts to change the political regime that draw on a competing vision of a just order, (b) a notable degree of informal or formal mass mobilization, and (c) efforts to force change through noninstitutionalized actions such as mass demonstrations, protests, strikes, or violence.".
Industrialisation (UK) or industrialization (US) is "the period of social and economic change that transforms a human group from an agrarian and feudal society into an industrial society. This involves an extensive reorganisation of an economy for the purpose of manufacturing." Industrialisation is associated with an increase in polluting industries heavily dependent on fossil fuels. With the increasing focus on sustainable development and green industrial policy practices, industrialisation increasingly includes technological leapfrogging, with direct investment in more advanced, cleaner technologies.
Nationalism is an ideology or movement that holds that the nation should be congruent with the state. As a movement, it presupposes the existence and tends to promote the interests of a particular nation, especially with the aim of gaining and maintaining its sovereignty (self-determination) over its perceived homeland to create a nation-state. It holds that the nation should govern itself, free from outside interference (self-governance), that a nation is a natural and ideal basis for a polity, and that the nation is the only rightful source of political power. It further aims to build, and maintain, a single national identity, based on a combination of shared social characteristics such as culture, ethnicity, homeland, language, politics, religion, traditions, or belief in a shared singular history, and to promote national unity or solidarity. There are various definitions of a "nation", which leads to different types of nationalism. The two main divergent forms are ethnic nationalism and civic nationalism.
Fascism is a far-right, authoritarian, and ultranationalist political ideology and movement that rose to prominence in early-20th-century Europe. Fascism is characterized by support for a dictatorial leader, centralized autocracy, militarism, forcible suppression of opposition, belief in a natural social hierarchy, subordination of individual interests for the perceived interest of the nation or race, and strong regimentation of society and the economy. Opposed to communism, democracy, liberalism, pluralism, and socialism, fascism is at the far-right of the traditional left–right spectrum. What constitutes a precise definition of fascism has been a longrunning and complex debate among scholars.
Communism is a political and economic ideology whose goal is the creation of a communist society, a socioeconomic order centered on common ownership of the means of production, distribution, and exchange that allocates products in society based on need. A communist society entails the absence of private property and social classes, and ultimately money and the state. Communism is a part of the broader socialist movement.
Capitalism is an economic system based on the private ownership of the means of production and its use for the purpose of obtaining profit. This socioeconomic system has developed historically in several stages, and is defined by a number of constituent elements: private property, profit motive, capital accumulation, competitive markets, commodification, wage labor, and an emphasis on innovation and economic growth. Capitalist economies may experience business cycles of economic expansion followed by recessions.
Feudalism, also known as the feudal system, was a combination of various customs and systems that flourished in medieval Europe from the 9th to 15th centuries. Broadly defined, it was a way of structuring society around relationships derived from the holding of land in exchange for service or labour.
Migration, migratory, or migrate may refer to:.
Slavery is the ownership of a person as property, especially in regard to their labour. It is an economic phenomenon and its history resides in economic history. Slavery typically involves compulsory work, with the slave's location of work and residence dictated by the party that holds them in bondage. Enslavement is the placement of a person into slavery, and the person is called a slave or an enslaved person.
Abolition refers to the act of putting an end to something by law, and may refer to:Abolitionism, abolition of slavery
Abolition of the death penalty, also called capital punishment
Abolition of monarchy
Abolition of nuclear weapons
Abolition of prisons
Abolition of ICE
Police abolition movement
Abolition of suffering
Abolitionism, related to veganism
Abolition of time zones
Abolition of borders.
Exploration is the process of exploring, an activity which has some expectation of discovery. Organised exploration is largely a human activity, but exploratory activity is common to most organisms capable of directed locomotion and the ability to learn, and has been described in, amongst others, social insects foraging behaviour, where feedback from returning individuals affects the activity of other members of the group.
Navigation is a field of study that focuses on the process of monitoring and controlling the movement of a craft or vehicle from one place to another. The field of navigation includes four general categories: land navigation, marine navigation, aeronautic navigation, and space navigation. It is also the term of art used for the specialized knowledge used by navigators to perform navigation tasks. All navigational techniques involve locating the navigator's position compared to known locations or patterns. Navigation, in a broader sense, can refer to any skill or study that involves the determination of position and direction. In this sense, navigation includes orienteering and pedestrian navigation.
Cartography is the study and practice of making and using maps. Combining science, aesthetics and technique, cartography builds on the premise that reality can be modeled in ways that communicate spatial information effectively.
Diplomacy is the communication by representatives of state, intergovernmental, or non-governmental institutions intended to influence events in the international system.
War is an armed conflict between the armed forces of states, or between governmental forces and armed groups that are organized under a certain command structure and have the capacity to sustain military operations, or between such organized groups.
Genocide is the partial or total destruction of a human group, committed intentionally. The popular view conceives of genocide as the large-scale killing of individuals, but in the scholarly and legal fields, genocide occurs when the group itself is targeted. Acts of genocide include killing as well as non-lethal acts such as preventing reproduction among the group, the forcible transfer of children to another group, and cultural genocide.
Propaganda is communication that is primarily used to influence or persuade an audience to further an agenda, which may not be objective and may be selectively presenting facts to encourage a particular synthesis or perception, or using loaded language to produce an emotional rather than a rational response to the information that is being presented. Propaganda can be found in a wide variety of different contexts.
Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research.
Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives.
Myth is a genre of folklore consisting primarily of narratives that play a fundamental role in a society. For scholars, this is totally different from the ordinary sense of the term myth, meaning a belief that is not true, as the veracity of a piece of folklore is entirely irrelevant to determining whether it constitutes a myth.
Trade involves the transfer of goods and services from one person or entity to another, often in exchange for money. Economists refer to a system or network that allows trade as a market.
Agriculture is the practice of cultivating the soil, planting, raising, and harvesting both food and non-food crops, as well as livestock production. Broader definitions also include forestry and aquaculture. Agriculture was a key factor in the rise of sedentary human civilization, whereby farming of domesticated plants and animals created food surpluses that enabled people to live in the cities. While humans started gathering grains at least 105,000 years ago, nascent farmers only began planting them around 11,500 years ago. Sheep, goats, pigs, and cattle were domesticated around 10,000 years ago. Plants were independently cultivated in at least 11 regions of the world. In the 20th century, industrial agriculture based on large-scale monocultures came to dominate agricultural output.
Urbanization is the population shift from rural to urban areas, the corresponding decrease in the proportion of people living in rural areas, and the ways in which societies adapt to this change. It can also mean population growth in urban areas instead of rural ones. It is predominantly the process by which towns and cities are formed and become larger as more people begin to live and work in central areas.
Globalization is the process of increasing interdependence and integration among the economies, markets, societies, and cultures of different countries worldwide. It can be attributed to a series of factors, including the reduction of barriers to international trade, the liberalization of capital movements, the development of transportation infrastructure, and the advancement of information and communication technologies. The term globalization first appeared in the early 20th century. It developed its current meaning sometime in the second half of the 20th century, and came into popular use in the 1990s to describe the unprecedented international connectivity of the post–Cold War world.
The origins of globalization can be traced back to the 18th and 19th centuries, a period marked by significant advancements in transportation and communication technologies. These developments increased global interactions, fostering the growth of international trade and the exchange of ideas, beliefs, and cultures. While globalization is primarily an economic process of interaction and integration, it is also closely linked to social and cultural dynamics. Additionally, disputes and international diplomacy have played significant roles in the history and evolution of globalization, continuing to shape its modern form. Though many scholars place the origins of globalization in modern times, others trace its history to long before the European Age of Discovery and voyages to the New World, and some even to the third millennium BCE. Large-scale globalization began in the 1820s, and in the late 19th century and early 20th century drove a rapid expansion in the connectivity of the world's economies and cultures. The term global city was subsequently popularized by sociologist Saskia Sassen in her work The Global City: New York, London, Tokyo (1991).
Independence is a condition of a nation, country, or state, in which residents and population, or some portion thereof, exercise self-government, and usually sovereignty, over its territory. The opposite of independence is the status of a dependent territory or colony. The commemoration of the independence day of a country or nation celebrates when a country is free from all forms of colonialism; free to build a country or nation without any interference from other nations.
Unification or unification theory may refer to:.


A civilization is any complex society characterized by the development of the state, social stratification, urbanization, and symbolic systems of communication beyond signed or spoken languages.
Prehistory, sometimes referred to as pre-literary history, is the period of human history between the first known use of stone tools by hominins c. 3.3 million years ago and the beginning of recorded history with the invention of writing systems. The use of symbols, marks, and images appears very early among humans, but the earliest known writing systems appeared c. 5,200 years ago. The adoption of writing across the globe has been a slow process, so that the end of prehistory occurred at different times in different places, and the term is less often used in discussing societies where prehistory ended relatively recently. The period when a culture is written about by others, but has not developed its own writing system, is often known as the protohistory of the culture.
Arithmetic is an elementary branch of mathematics that deals with numerical operations like addition, subtraction, multiplication, and division. In a wider sense, it also includes exponentiation, extraction of roots, and taking logarithms.
Algebra is a branch of mathematics that deals with abstract systems, known as algebraic structures, and the manipulation of expressions within those systems. It is a generalization of arithmetic that introduces variables and algebraic operations other than the standard arithmetic operations, such as addition and multiplication.
Geometry is a branch of mathematics concerned with properties of space such as the distance, shape, size, and relative position of figures. Geometry is, along with arithmetic, one of the oldest branches of mathematics. A mathematician who works in the field of geometry is called a geometer. Until the 19th century, geometry was almost exclusively devoted to Euclidean geometry, which includes the notions of point, line, plane, distance, angle, surface, and curve, as fundamental concepts.
Trigonometry is a branch of mathematics concerned with relationships between angles and side lengths of triangles. In particular, the trigonometric functions relate the angles of a right triangle with ratios of its side lengths. The field emerged in the Hellenistic world during the 3rd century BC from applications of geometry to astronomical studies. The Greeks focused on the calculation of chords, while mathematicians in India created the earliest-known tables of values for trigonometric ratios such as sine.
Calculus is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations.
Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments.
Probability concerns events and numerical descriptions of how likely they are to occur. The probability of an event is a number between 0 and 1; the larger the probability, the more likely an event is to occur. This number is often expressed as a percentage (%), ranging from 0% to 100%. A simple example is the tossing of a fair (unbiased) coin. Since the coin is fair, the two outcomes are both equally probable; the probability of "heads" equals the probability of "tails"; and since no other outcomes are possible, the probability of either "heads" or "tails" is 1/2.
Combinatorics is an area of mathematics primarily concerned with counting, both as a means and as an end to obtaining results, and certain properties of finite structures. It is closely related to many other areas of mathematics and has many applications ranging from logic to statistical physics and from evolutionary biology to computer science.
Graph may refer to:.
Logic is the study of correct reasoning. It includes both formal and informal logic. Formal logic is the study of deductively valid inferences or logical truths. It examines how conclusions follow from premises based on the structure of arguments alone, independent of their topic and content. Informal logic is associated with informal fallacies, critical thinking, and argumentation theory. Informal logic examines arguments expressed in natural language whereas formal logic uses formal language. When used as a countable noun, the term "a logic" refers to a specific logical formal system that articulates a proof system. Logic plays a central role in many fields, such as philosophy, mathematics, computer science, and linguistics.
Set, The Set, SET or SETS may refer to:.
Analysis is the process of breaking a complex topic or substance into smaller parts in order to gain a better understanding of it. The technique has been applied in the study of mathematics and logic since before Aristotle, though analysis as a formal concept is a relatively recent development.
Mathematical optimization or mathematical programming is the selection of a best element, with regard to some criteria, from some set of available alternatives. It is generally divided into two subfields: discrete optimization and continuous optimization. Optimization problems arise in all quantitative disciplines from computer science and engineering to operations research and economics, and the development of solution methods has been of interest in mathematics for centuries.
Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications.
Symmetry in everyday life refers to a sense of harmonious and beautiful proportion and balance. In mathematics, the term has a more precise definition and is usually used to refer to an object that is invariant under some transformations, such as translation, reflection, rotation, or scaling. Although these two meanings of the word can sometimes be told apart, they are intricately related, and hence are discussed together in this article.
In mathematics, a fractal is a geometric shape containing detailed structure at arbitrarily small scales, usually having a fractal dimension strictly exceeding the topological dimension. Many fractals appear similar at various scales, as illustrated in successive magnifications of the Mandelbrot set. This exhibition of similar patterns at increasingly smaller scales is called self-similarity, also known as expanding symmetry or unfolding symmetry; if this replication is exactly the same at every scale, as in the Menger sponge, the shape is called affine self-similar. Fractal geometry relates to the mathematical branch of measure theory by their Hausdorff dimension.
Matrix or MATRIX may refer to:.
Vector most often refers to:Disease vector, an agent that carries and transmits an infectious pathogen into another living organism
Euclidean vector, a quantity with a magnitude and a direction.
Function or functionality may refer to:.
The derivative of a function is the rate of change of the function's output relative to its input value.
In mathematics, an integral is the continuous analog of a sum, and is used to calculate areas, volumes, and their generalizations. The process of computing an integral, called integration, is one of the two fundamental operations of calculus, along with differentiation. Integration was initially used to solve problems in mathematics and physics, such as finding the area under a curve, or determining displacement from velocity. Usage of integration expanded to a wide variety of scientific fields thereafter.
Limit or Limits may refer to:.
Series may refer to:.
In mathematics, an equation is a mathematical formula that expresses the equality of two expressions, by connecting them with the equals sign =. The word equation and its cognates in other languages may have subtly different meanings; for example, in French an équation is defined as containing one or more variables, while in English, any well-formed formula consisting of two expressions related with an equals sign is an equation.
Inequality may refer to:Inequality (mathematics), a relation between two quantities when they are different.
Economic inequality, difference in economic well-being between population groups
Income inequality, an unequal distribution of income
Wealth inequality, an unequal distribution of wealth
Spatial inequality, the unequal distribution of income and resources across geographical regions
International inequality, economic differences between countries
Social inequality, unequal opportunities and rewards for different social positions or statuses within a group
Gender inequality, unequal treatment or perceptions due to gender
Racial inequality, social distinctions between racial and ethnic groups within a society
Health inequality, differences in the quality of health and healthcare across populations
Educational inequality, the unequal distribution of academic resources
Environmental inequality, unequal environmental harms between different neighborhoods or cities
Urban forest inequity, an unequal distribution of trees
Attention inequality, unequal distribution of attention across users, groups of people, issues in etc. in attention economy
Participation inequality, the phenomenon in which a small percentage of people contributes the majority of information to the total outcome.
Transformation may refer to:.
In mathematics and computer science, an algorithm is a finite sequence of mathematically rigorous instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations and data processing. More advanced algorithms can use conditionals to divert the code execution through various routes and deduce valid inferences.
Computer numerical control (CNC) or CNC machining is the automated control of machine tools by a computer. It is an evolution of numerical control (NC), where machine tools are directly managed by data storage media such as punched cards or punched tape. Because CNC allows for easier programming, modification, and real-time adjustments, it has gradually replaced NC as computing costs declined.
Dynamics or dynamic may refer to:.
Chaos or CHAOS may refer to:.
In mathematics, a manifold is a topological space that locally resembles Euclidean space near each point. More precisely, an -dimensional manifold, or -manifold for short, is a topological space with the property that each point has a neighborhood that is homeomorphic to an open subset of -dimensional Euclidean space.
In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects associated with a vector space. Tensors may map between different objects such as vectors, scalars, and even other tensors. There are many types of tensors, including scalars and vectors, dual vectors, multilinear maps between vector spaces, and even some operations such as the dot product. Tensors are defined independent of any basis, although they are often referred to by their components in a basis related to a particular coordinate system; those components form an array, which can be thought of as a high-dimensional matrix.
Measure may refer to:.
In mathematics, cardinality is an inherent property of sets, roughly meaning the number of individual objects they contain, which may be infinite. The concept is understood through one-to-one correspondences between sets. That is, if their objects can be paired such that each object has a pair, and no object is paired more than once.
Infinity is something which is boundless, limitless, endless. It is denoted by ∞, called the infinity symbol.
Modularity is the degree to which a system's components may be separated and recombined, often with the benefit of flexibility and variety in use. The concept of modularity is used primarily to reduce complexity by breaking a system into varying degrees of interdependence and independence across and "hide the complexity of each part behind an abstraction and interface". However, the concept of modularity can be extended to multiple disciplines, each with their own nuances. Despite these nuances, consistent themes concerning modular systems can be identified.
In mathematics, a polynomial is a mathematical expression consisting of indeterminates and coefficients, that involves only the operations of addition, subtraction, multiplication and exponentiation to nonnegative integer powers, and has a finite number of terms. An example of a polynomial of a single indeterminate  is . An example with three indeterminates is .
In mathematics, factorization (or factorisation, see English spelling differences) or factoring consists of writing a number or another mathematical object as a product of several factors, usually smaller or simpler objects of the same kind. For example, 3 × 5 is an integer factorization of 15, and (x − 2)(x + 2) is a polynomial factorization of x2 − 4.
A computation is any type of arithmetic or non-arithmetic calculation that is well-defined. Common examples of computation are mathematical equation solving and the execution of computer algorithms.
Complexity characterizes the behavior of a system or model whose components interact in multiple ways and follow local rules, leading to non-linearity, randomness, collective dynamics, hierarchy, and emergence.
Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
Operator may refer to:.
In linear algebra, an eigenvector or characteristic vector is a (nonzero) vector that has its direction unchanged by a given linear transformation. More precisely, an eigenvector  of a linear transformation  is scaled by a constant factor  when the linear transformation is applied to it: . The corresponding eigenvalue, characteristic value, or characteristic root is the multiplying factor .
In linear algebra, an eigenvector or characteristic vector is a (nonzero) vector that has its direction unchanged by a given linear transformation. More precisely, an eigenvector  of a linear transformation  is scaled by a constant factor  when the linear transformation is applied to it: . The corresponding eigenvalue, characteristic value, or characteristic root is the multiplying factor .
Stochastic is the property of being well-described by a random probability distribution. Stochasticity and randomness are technically distinct concepts. Stochasticity refers to a modeling approach, while randomness describes phenomena. These terms are often used interchangeably. In probability theory, the formal concept of a stochastic process is also referred to as a random process.
Regression or regressions may refer to:.
A model is an informative representation of an object, person, or system. The term originally denoted the plans of a building in late 16th-century English, and derived via French and Italian ultimately from Latin modulus, 'a measure'.
Proof most often refers to:Proof (truth), argument or sufficient evidence for the truth of a proposition
Alcohol proof, a measure of an alcoholic drink's strength.
In mathematics and formal logic, a theorem is a statement that has been proven, or can be proven. The proof of a theorem is a logical argument that uses the inference rules of a deductive system to establish that the theorem is a logical consequence of the axioms and previously proved theorems.
Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis).
Chemistry is the scientific study of the properties and behavior of matter. It is a physical science within the natural sciences that studies the chemical elements that make up matter and compounds made of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during reactions with other substances. Chemistry also addresses the nature of chemical bonds in chemical compounds.
Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist.
Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole.
Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science.
Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history.
Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene.
Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences.
Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Economics is a social science that studies the production, distribution, and consumption of goods and services.
Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions.
Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics.
Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions.
Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role.
Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty.
Music is the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content. Music is generally agreed to be a cultural universal that is present in all human societies. Definitions of music vary widely in substance and approach. While scholars agree that music is defined by a small number of specific elements, there is no consensus as to what these necessary elements are. Music is often characterized as a highly versatile medium for expressing human creativity. Diverse activities are involved in the creation of music, and are often divided into categories of composition, improvisation, and performance. Music may be performed using a wide variety of musical instruments, including the human voice. It can also be composed, sequenced, or otherwise produced to be indirectly played mechanically or electronically, such as via a music box, barrel organ, or digital audio workstation software on a computer.
Theatre or theater is a collaborative form of performing art that uses live performers, usually actors, to present experiences of a real or imagined event before a live audience in a specific place, often a stage. The performers may communicate this experience to the audience through combinations of gesture, speech, song, music, and dance. It is the oldest form of drama, though live theatre has now been joined by modern recorded forms. Elements of art, such as painted scenery and stagecraft such as lighting are used to enhance the physicality, presence and immediacy of the experience. Places, normally buildings, where performances regularly take place are also called "theatres", as derived from the Ancient Greek θέατρον, itself from θεάομαι.
A film, movie, or motion picture is a work of visual art that simulates experiences and otherwise communicates ideas, stories, perceptions, emotions, or atmosphere through the use of moving images that are generally, since the 1930s, synchronized with sound and sometimes using other sensory stimuli.
Media may refer to:.
Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources.
The branch of social science that studies politics and government is referred to as political science.
Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people.
Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.".
Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure, machinery, vehicles, electronics, materials, and energy systems. For related terms, see glossary of engineering.
Medicine is the science and practice of caring for patients, managing the diagnosis, prognosis, prevention, treatment and palliation of their injury or disease, while promoting their health. Medicine encompasses a variety of health care practices which evolved to maintain and restore health through the prevention and treatment of illness. Contemporary medicine applies biomedical sciences, biomedical research, genetics, and medical technology to diagnose, treat, and prevent injury and disease, typically through various pharmaceuticals or surgery, but also through therapies such as psychotherapy, external splints and traction, medical devices, biologics, and ionizing radiation, amongst others.
Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. Robotics usually combines four aspects of design work: a power source, mechanical construction, a control system, and software.
Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide.
A design is the concept or proposal for an object, process, or system. The word design refers to something that is or has been intentionally created by a thinking agent, and is sometimes used to refer to the inherent nature of something – its design. The verb to design expresses the process of developing a design. In some cases, the direct construction of an object without an explicit prior plan may also be considered to be a design, such as in arts and crafts. A design is expected to have a purpose within a specific context, typically aiming to satisfy certain goals and constraints while taking into account aesthetic, functional and experiential considerations. Traditional examples of designs are architectural and engineering drawings, circuit diagrams, sewing patterns, and less tangible artefacts such as business process models.
Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction. It is both the process and the product of sketching, conceiving, planning, designing, and constructing buildings or other structures. The term comes from Latin  architectura; from Ancient Greek  ἀρχιτέκτων (arkhitéktōn) 'architect'; from  ἀρχι- (arkhi-) 'chief' and  τέκτων (téktōn) 'creator'. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilizations are often identified with their surviving architectural achievements.
Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena.
A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs, which enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation, or to a group of computers that are linked and function together, such as a computer network or computer cluster.
An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions.
Google LLC is an American multinational technology corporation focused on information technology, online advertising, search engine technology, email, cloud computing, software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI). It has been referred to as "the most powerful company in the world" by the BBC, and is one of the world's most valuable brands. Google's parent company Alphabet Inc. has been described as a Big Tech company.
Microsoft Corporation is an American multinational technology conglomerate headquartered in Redmond, Washington. Founded in 1975, the company became influential in the rise of personal computers through software like Windows, and has since expanded to Internet services, cloud computing, artificial intelligence, video gaming, and other fields. Often described as a Big Tech company, Microsoft is the largest software company by revenue, one of the most valuable public companies, and one of the most valuable brands globally. It is a part of Big Tech along with five other tech companies in the United States, Alphabet (Google), Amazon, Apple, Meta (Facebook), and Nvidia, which are also the largest companies in the world by market capitalization.
Amazon most often refers to:Amazon (company), an American multinational technology company
Amazon rainforest, a rainforest covering most of the Amazon basin
Amazon River, in South America
Amazons, a tribe of female warriors in Greek mythology.
Meta most commonly refers to:Meta (prefix), a common affix and word in English
Meta Platforms, an American multinational technology conglomerate .
Tesla most commonly refers to:Nikola Tesla (1856–1943), a Serbian-American electrical engineer and inventor
Tesla, Inc., an American electric vehicle and clean energy company, formerly Tesla Motors, Inc.
Tesla (unit), the SI-derived unit of magnetic flux density.
Nvidia Corporation is an American technology company headquartered in Santa Clara, California. Founded on April 5, 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops graphics processing units (GPUs), systems on chips (SoCs), and application programming interfaces (APIs) for data science, high-performance computing, video games, and mobile and automotive applications. Nvidia has been described as a Big Tech company.
Samsung Group is a South Korean multinational manufacturing conglomerate headquartered in the Samsung Town office complex in Seoul. The group consists of numerous affiliated businesses, most of which operate under the Samsung brand, and is the largest chaebol in South Korea. As of 2024, Samsung has the world's fifth-highest brand value.
Intel Corporation is an American multinational technology company headquartered in Santa Clara, California. It designs, manufactures, and sells computer components such as central processing units (CPUs) and related products for business and consumer markets. Intel was the world's third-largest semiconductor chip manufacturer by revenue in 2024 and has been included in the Fortune 500 list of the largest United States corporations by revenue since 2007. It was one of the first companies listed on Nasdaq.
International Business Machines Corporation, doing business as IBM, is an American multinational technology company headquartered in Armonk, New York, and present in over 175 countries. It is a publicly traded company and one of the 30 companies in the Dow Jones Industrial Average. IBM is the largest industrial research organization in the world, with 19 research facilities across a dozen countries; for 29 consecutive years, from 1993 to 2021, it held the record for most annual U.S. patents generated by a business.
An oracle is a person or thing considered to provide insight, wise counsel or prophetic predictions, most notably including precognition of the future, inspired by deities. If done through occultic means, it is a form of divination.
Cisco Systems, Inc. is an American multinational technology conglomerate corporation that develops, manufactures, and sells hardware, software, telecommunications equipment and other high-technology services and products focused on networking, cyber security and AI. Cisco specializes in specific tech markets, such as the Internet of things (IoT), domain security, videoconferencing, and energy management, including products such as Webex, OpenDNS, Jabber, and Jasper. The company is headquartered in San Jose, California and, as of December 2025, has a market capitalization of $317 billion.
Adobe is a building material made from loam and organic materials. Adobe is Spanish for mudbrick. In some English-speaking regions of Spanish heritage, such as the Southwestern United States, the term is used to refer to any kind of earthen construction, or various architectural styles like Pueblo Revival or Territorial Revival. Most adobe buildings are similar in appearance to cob and rammed earth buildings. Adobe is among the earliest building materials, and is used throughout the world.
Spotify is a Swedish audio streaming and media service provider founded in April 2006 by Daniel Ek and Martin Lorentzon. As of December 2025, it was one of the largest providers of music streaming services, with over 751 million monthly active users comprising 290 million paying subscribers. Spotify is listed on the New York Stock Exchange in the form of American depositary receipts.
Netflix is an American subscription video on-demand over-the-top streaming service. The service primarily distributes original and acquired films and television shows from various genres. It is available internationally in multiple languages.
The Walt Disney Company, commonly known as simply Disney, is an American multinational mass media and entertainment conglomerate headquartered at the Walt Disney Studios complex in Burbank, California. Founded on October 16, 1923, as an animation studio by brothers Walt Disney and Roy Oliver Disney as Disney Brothers Cartoon Studio, Disney operated under the names Walt Disney Studio and Walt Disney Productions before adopting its current name in 1986. In 1928, Disney established itself as a leader in the animation industry with the short film Steamboat Willie. The film used synchronized sound to become the first post-produced sound cartoon, and popularized Mickey Mouse, who became Disney's mascot and corporate icon.
Sony Group Corporation, commonly referred to as Sony, is a Japanese multinational conglomerate headquartered at Sony City in Minato, Tokyo, Japan. The Sony Group encompasses various businesses, including electronics, imaging and sensing, film and television, music, video games, and others.
Nintendo Co., Ltd. is a Japanese multinational video game company headquartered in Kyoto. It develops, publishes, and manufactures both video games and video game consoles.
Uber Technologies, Inc. is an American multinational transportation company that provides ride-hailing services, courier services, food delivery, and freight transport. It is headquartered in San Francisco, California, and operates in approximately 70 countries and 15,000 cities worldwide. It is the largest ridesharing company worldwide with over 202 million monthly active users and 10 million active drivers and couriers. It coordinates an average of 42 million trips and delivery orders per day, and has coordinated 72 billion trips and delivery orders since its inception in 2010. In the fourth quarter of 2025, the company had a take rate of 29.9% for mobility services and 19.2% for food delivery.
Lyft, Inc. is an American company offering ride-hailing services, motorized scooters, and bicycle-sharing systems in the United States and Canada, and, via its Free Now mobile app, Europe. Lyft is the second-largest ridesharing company in the United States after Uber. It has 25 million active riders and coordinates 9 million rides per day.
Airbnb, Inc. is an American company operating an online marketplace for short-and-long-term homestays, experiences and services in various countries and regions. It acts as a broker and charges a commission from each booking. Airbnb was founded in 2008 by Brian Chesky, Nathan Blecharczyk, and Joe Gebbia.
Stripe, striped, or stripes may refer to:.
In geometry, a square is a regular quadrilateral. It has four straight sides of equal length and four equal angles. Squares are special cases of rectangles, which have four equal angles, and of rhombuses, which have four equal sides. As with all rectangles, a square's angles are right angles, making adjacent sides perpendicular. The area of a square is the side length multiplied by itself, and so in algebra, multiplying a number by itself is called squaring.
PayPal Holdings, Inc. is an American multinational financial technology company operating an online payments system in the majority of countries that support online money transfers; it serves as an electronic alternative to traditional paper methods such as checks and money orders. The company operates as a payment processor for online vendors, auction sites and many other commercial and company users.
Visa most commonly refers to:Travel visa, a document allowing entry to a foreign country
Work visa, a document granting permission to work in a foreign country
Visa Inc., a US multinational financial and payment cards company
Visa Debit card issued by the above company
Visa Plus, an interbank network
Visa Electron, a debit card.
Mastercard Inc. is an American multinational payment card services corporation headquartered in Purchase, New York. It offers a range of payment transaction processing and other related-payment services. Throughout the world, its principal business is to process payments between the banks of merchants and the card-issuing banks or credit unions of the purchasers who use the Mastercard-brand debit, credit and prepaid cards to make purchases. Mastercard has been publicly traded since 2006.
Coca-Cola, or Coke, is a cola soft drink manufactured by the Coca-Cola Company. In 2013, Coke products were sold in over 200 countries and territories worldwide, with consumers drinking more than 1.8 billion company beverage servings each day. Coca-Cola ranked No. 94 in the 2024 Fortune 500 list of the largest United States corporations by revenue. Based on Interbrand's "best global brand" study of 2023, Coca-Cola was the world's sixth most valuable brand.
Pepsi is a carbonated soft drink with a cola flavor, manufactured by PepsiCo which serves as its flagship product. In 2023, Pepsi was the second most valuable soft drink brand worldwide behind Coca-Cola; the two share a long-standing rivalry in what has been called the "cola wars".
Nike often refers to:Nike, Inc., a major American producer of athletic shoes, apparel, and sports equipment
Nike (mythology), a Greek goddess who personifies victory.
Adidas AG is a German multinational athletic apparel and footwear corporation headquartered in Herzogenaurach, Germany. It is the largest sportswear manufacturer in Europe, and the second largest in the world, after Nike. It is the holding company for the Adidas Group, which also owns an 8.33% stake in the football club Bayern Munich, and Runtastictrian fitness technology company. Adidas's revenue for 2024 was listed at €23 billion. Adidas is best known for their iconic brand image, offering the Yeezy Boost sneakers, and is publicly recognized for their extensive long origin history for participating in sponsored athletes, and for providing gear in the FIFA World Cup series. The brand is also unique for performance innovation of their shoes with major deep ties with sports culture, and durability with their focus of sport shoes, clothing, backpacks, and other accessories. Its commitment to sustainability includes their digital technology and AI, including collaborating with cultural figures like Lionel Messi, Patrick Mahomes, Real Madrid, and Pharrell Williams.
Puma or PUMA may refer to:.
Toyota Motor Corporation  is a Japanese multinational automotive manufacturer headquartered in Toyota City, Aichi, Japan. It was founded by Kiichiro Toyoda and incorporated on August 28, 1937. Toyota is the largest automobile manufacturer in the world, producing about 10 million vehicles per year.
Honda Motor Co., Ltd., commonly known as Honda, is a Japanese multinational conglomerate automotive manufacturer headquartered at the Toranomon Alcea Tower in Toranomon, Minato, Tokyo, Japan.
Ford commonly refers to:Ford Motor Company, an automobile manufacturer founded by Henry Ford
Ford (crossing), a shallow crossing on a river.
Bayerische Motoren Werke Aktiengesellschaft, trading as BMW Group, is a German multinational conglomerate manufacturer of luxury vehicles and motorcycles headquartered in Munich, Germany. The moniker, "BMW ", first came into use when the German firm Rapp Motorenwerke changed its name to Bayerische Motoren Werke GmbH in 1917. Thereafter, in 1922, the name and assets of BMW GmbH were transferred to the aircraft manufacturer Bayerische Flugzeugwerke AG, thereby giving rise to the company known today as BMW AG.
Mercedes may refer to:.
Volkswagen is a German automobile manufacturer based in Wolfsburg, Lower Saxony, Germany. Established in 1937 by the German Labour Front, it was revived after World War II by British Army officer Ivan Hirst and over the 81 years since grew into the global brand it is today. The company is well known for the Beetle and serves as the flagship marque of the Volkswagen Group, which was the world's largest automotive manufacturer by global sales in 2016 and 2017.
Shell or Shells may refer to:.
Exxon Mobil Corporation is an American multinational oil and gas corporation headquartered in Spring, Texas, a suburb of Houston. Founded as the largest direct successor of John D. Rockefeller's Standard Oil, the company was formed in 1999, with the merger of Exxon and Mobil. It is vertically integrated across the entire oil and gas industry, as well as within its chemicals division, which produces plastic, synthetic rubber, and other chemical products. As the largest U.S.-based oil and gas company, ExxonMobil is the eighth-largest company by revenue in the U.S. and 13th-largest in the world. It is also the largest investor-owned oil company in the world. Approximately 55.56% of the company's shares are held by institutions, the largest of which, as of 2019, were The Vanguard Group (8.15%), BlackRock (6.61%), and State Street Corporation (4.83%).
Chevron may refer to:.
Walmart Inc. is an American multinational retail corporation that operates a chain of hypermarkets, discount department stores, and grocery stores in the United States and 19 other countries. It is headquartered in Bentonville, Arkansas. The company was founded in 1962 by brothers Sam Walton and James "Bud" Walton in nearby Rogers, Arkansas. It also owns and operates Sam's Club retail warehouses.
Target may refer to:.
Costco Wholesale Corporation is an American multinational corporation that operates a chain of membership-only big-box warehouse club retail stores. As of 2021, Costco is the third-largest retailer in the world, and as of August 2024, the world's largest retailer of beef, poultry, organic produce, and wine, with just under a third of American consumers regularly shopping at Costco warehouses. As of 2025, Costco is ranked 12th on the Fortune 500 rankings of the largest United States corporations by total revenue.
IKEA is a multinational conglomerate founded in Sweden that designs and sells ready-to-assemble furniture, household goods, and various related services.
Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington. It was founded in 1971 by Jerry Baldwin, Zev Siegl, and Gordon Bowker at Seattle's Pike Place Market initially as a coffee bean wholesaler. Starbucks was converted into a coffee shop serving espresso-based drinks under the ownership of Howard Schultz, who was chief executive officer from 1986 to 2000 and led the aggressive expansion of the franchise across the West Coast of the United States.
McDonald's Corporation, doing business as McDonald's, is an American multinational fast food restaurant chain. As of 2024, it is the second-largest by number of locations in the world, behind the Chinese chain Mixue Ice Cream & Tea.
A chipotle, or chilpotle, is a smoke-dried ripe jalapeño chili pepper used for seasoning. It is used primarily in Mexican and Mexican-inspired cuisines, such as Tex-Mex and Southwestern United States dishes. It comes in different forms, such as chipotles en adobo.
Dominoes is a family of tile-based games played with pieces. Each domino is a rectangular tile, usually with a line dividing its face into two square ends. Each end is marked with a number of spots or is blank. The backs of the tiles in a set are indistinguishable, either blank or having some common design. The gaming pieces make up a domino set, sometimes called a deck or pack. The traditional European domino set consists of 28 tiles, also known as pieces, bones, rocks, stones, men, cards or just dominoes, featuring all combinations of spot counts between zero and six. A domino set is a generic gaming device, similar to playing cards or dice, in that a variety of games can be played with a set. Another form of entertainment using domino pieces is the practice of domino toppling.
FedEx Corporation, originally known as Federal Express Corporation, is an American multinational conglomerate holding company specializing in transportation, e-commerce, and business services. The company is headquartered in Memphis, Tennessee. The name "FedEx" is a syllabic abbreviation of its original air division, Federal Express, which operated under this name from 1973 until 1994.
UPS commonly refers to:Uninterruptible power supply, a device which provides continuous power to electronics
United Parcel Service, an American courier company.

Abstraction is the process of generalizing rules and concepts from specific examples, literal signifiers, first principles, or other methods. The result of the process, an abstraction, is a concept that acts as a common noun for all subordinate concepts and connects any related concepts as a group, field, or category.
In mechanics, an acceleration is a change in velocity and is calculated as the rate of change of the velocity of an object with respect to time. Acceleration is apart of the study of motion and is one of several components of kinematics. Acceleration has magnitude and direction, making it a vector quantity. Fundamentally, an acceleration is any time an object changes speed or direction.
Acoustics is a branch of continuum mechanics that deals with the study of mechanical waves in gases, liquids, and solids including topics such as vibration, sound, ultrasound and infrasound. A scientist who works in the field of acoustics is an acoustician while someone working in the field of acoustics technology may be called an acoustical engineer. The application of acoustics is present in almost all aspects of modern society with the most obvious being the audio and noise control industries.
In biology, adaptation has three related meanings. Firstly, it is the dynamic evolutionary process of natural selection that fits organisms to their environment, enhancing their evolutionary fitness. Secondly, it is a state reached by the population during that process. Thirdly, it is a phenotypic trait or adaptive trait, with a functional role in each individual organism, that is maintained and has evolved through natural selection.
Adhesion is the tendency of dissimilar particles or surfaces to cling to one another.
Aeronautics is the science or art involved with the study, design, and manufacturing of air flight-capable machines, and the techniques of operating aircraft and rockets within the atmosphere.
While the term originally referred solely to operating the aircraft, it has since been expanded to include technology, business, and other aspects related to aircraft. The term "aviation" is sometimes used interchangeably with aeronautics, although "aeronautics" includes lighter-than-air craft such as airships, and includes ballistic vehicles while "aviation" technically does not.
An aerosol is a suspension of fine solid particles or liquid droplets in air or another gas. Aerosols can be generated from natural or human causes. The scientific term aerosol refers to the mixture of particulates in gas, and not to the particulate matter alone. The liquid or solid particles in an aerosol have diameters typically less than 1 μm. Larger particles with a significant settling speed make the mixture a suspension, although the distinction is not clear.
Affinity may refer to:.
Agronomy is the science and technology of producing and using plants by agriculture for food, fuel, fiber, chemicals, recreation, or land conservation. Agronomy has come to include research of plant genetics, plant physiology, meteorology, and soil science. It is the application of a combination of sciences such as biology, chemistry, economics, ecology, earth science, and genetics. Professionals in the field are known as agronomists.
In mathematics and computer science, an algorithm is a finite sequence of mathematically rigorous instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations and data processing. More advanced algorithms can use conditionals to divert the code execution through various routes and deduce valid inferences.
Alkalinity (from Arabic: القلوية, romanized: al-qaly, lit. 'ashes of the saltwort') is the capacity of water to resist acidification. It should not be confused with basicity, which is an absolute measurement on the pH scale. Alkalinity is the strength of a buffer solution composed of weak acids and their conjugate bases. It is measured by titrating the solution with an acid such as HCl until its pH changes abruptly, or it reaches a known endpoint where that happens. Alkalinity is expressed in units of concentration, such as meq/L (milliequivalents per liter), μeq/kg (microequivalents per kilogram), or mg/L CaCO3 (milligrams per liter of calcium carbonate). Each of these measurements corresponds to an amount of acid added as a titrant.
An alloy is a mixture of chemical elements of which in most cases at least one is a metallic element, although it is also sometimes used for mixtures of elements; herein only metallic alloys are described. Metallic alloys often have properties that differ from those of the pure elements from which they are made. The vast majority of metals used for commercial purposes are alloyed to improve their properties or behavior, such as increased strength, hardness or corrosion resistance. Metals may also be alloyed to reduce their overall cost, for instance alloys of gold and copper.
Altitude is a distance measurement, usually in the vertical or "up" direction, between a reference datum and a point or object. The exact definition and reference datum varies according to the context. Although the term altitude is commonly used to mean the height above sea level of a location, in geography the term elevation is often preferred for this usage.
Amphibians are ectothermic, anamniotic, four-limbed vertebrate animals that constitute the class Amphibia. In its broadest sense, it is a paraphyletic group encompassing all tetrapods, but excluding the amniotes. All extant (living) amphibians belong to the monophyletic subclass Lissamphibia, with three living orders: Anura, Urodela (salamanders), and Gymnophiona (caecilians). Evolved to be mostly semiaquatic, amphibians have adapted to inhabit a wide variety of habitats, with most species living in freshwater, wetland or terrestrial ecosystems. Their life cycle typically starts out as aquatic larvae with gills known as tadpoles, but some species have developed behavioural adaptations to bypass this.
The amplitude of a periodic variable is a measure of its change in a single period. The amplitude of a non-periodic signal is its magnitude compared with a reference value. There are various definitions of amplitude, which are all functions of the magnitude of the differences between the variable's extreme values. In older texts, the phase of a periodic function is sometimes called the amplitude.
Anatomy is the branch of morphology concerned with the study of the internal and external structure of organisms and their parts. Anatomy is a branch of natural science that deals with the structural organization of living things. It is an old science, having its beginnings in prehistoric times.
In meteorology, an anemometer is a device that measures wind speed and direction. It is a common instrument used in weather stations. The earliest known description of an anemometer was by Italian architect and author Leon Battista Alberti (1404–1472) in 1450.
Angiogenesis is the physiological process through which new blood vessels form from pre-existing vessels, formed in the earlier stage of vasculogenesis. Angiogenesis continues the growth of the vasculature mainly by processes of sprouting and splitting, but processes such as coalescent angiogenesis, vessel elongation and vessel cooption also play a role. Vasculogenesis is the embryonic formation of endothelial cells from mesoderm cell precursors, and from neovascularization, although discussions are not always precise. The first vessels in the developing embryo form through vasculogenesis, after which angiogenesis is responsible for most, if not all, blood vessel growth during development and in disease.
Anomaly, The Anomaly or Anomalies may refer to:.
An anode usually is an electrode of a polarized electrical device through which conventional current enters the device. This contrasts with a cathode, which is usually an electrode of the device through which conventional current leaves the device. A common mnemonic is ACID, for anode current into device. The direction of conventional current in a circuit is opposite to the direction of electron flow, so electrons flow from the anode of a galvanic cell, into an outside or external circuit connected to the cell. For example, the end of a household battery marked with a + is the cathode.
An antibody (Ab), or immunoglobulin (Ig), is a large protein belonging to the immunoglobulin superfamily which is used by the immune system to identify and neutralize antigens such as those that exist on bacteria and virus cells, including those that cause disease. Each individual antibody recognizes one or more specific antigens, and antigens of virtually any size and chemical composition can be recognized. Each of the branching chains comprising the "Y" of an antibody contains a paratope that specifically binds to one particular epitope on an antigen, allowing the two molecules to bind together with precision. Using this mechanism, antibodies can effectively "tag" the antigen for attack by cells of the immune system, or can neutralize it directly.
In immunology, an antigen (Ag) is a molecule, or portion thereof, that can bind to a specific antibody or T-cell receptor. The presence of antigens in the body may trigger an immune response.
In modern physics, antimatter is defined as matter composed of the antiparticles of the corresponding particles in "ordinary" matter, and can be thought of as matter with reversed charge and parity, or going backward in time. Antimatter occurs in natural processes like cosmic ray collisions and some types of radioactive decay, but only a tiny fraction of these have successfully been bound together in experiments to form antiatoms. Minuscule numbers of antiparticles can be generated at particle accelerators, but total artificial production has been only a few nanograms. No macroscopic amount of antimatter has ever been assembled due to the extreme cost and difficulty of production and handling. Nonetheless, antimatter is an essential component of widely available applications related to beta decay, such as positron emission tomography, radiation therapy, and industrial imaging.
In optics, the aperture of an optical system is the hole or opening that primarily limits light propagated through the system. The aperture defines a bundle of rays from each point on an object that will come to a focus in the image plane.
Apoptosis is a form of programmed cell death that occurs in multicellular organisms and in some eukaryotic, single-celled microorganisms such as yeast. Biochemical events lead to characteristic cell changes (morphology) and death. These changes include blebbing, cell shrinkage, nuclear fragmentation, chromatin condensation, DNA fragmentation, and mRNA decay. The average adult human loses 50 to 70 billion cells each day to apoptosis. For the average human child between 8 and 14 years old, each day the approximate loss is 20 to 30 billion cells.
Aquaculture, also known as aquafarming, is the controlled cultivation ("farming") of aquatic organisms such as fish, crustaceans, mollusks, algae and other organisms of value such as aquatic plants. Aquaculture involves cultivating freshwater, brackish water, and saltwater populations under controlled or semi-natural conditions and can be contrasted with commercial fishing, which is the harvesting of wild fish. Aquaculture is also a practice used for restoring and rehabilitating marine and freshwater ecosystems. Mariculture, commonly known as marine farming, is aquaculture in seawater habitats and lagoons, as opposed to freshwater aquaculture. Pisciculture is a type of aquaculture that consists of fish farming to obtain fish products as food.
Arbitrage is the practice of taking advantage of a difference in prices in two or more markets – striking a combination of matching deals to capitalize on the difference, the profit being the difference between the market prices at which the unit is traded. Arbitrage has the effect of causing prices of the same or very similar assets in different markets to converge.
An arboretum is a botanical collection composed exclusively of trees and shrubs of a variety of species. Originally mostly created as a section in a larger garden or park for specimens of mostly non-local species, many modern arboreta are in botanical gardens as living collections of woody plants and are intended at least in part for scientific study.
Archaea is a domain of organisms. Traditionally, Archaea included only its prokaryotic members, but has since been found to be paraphyletic, as eukaryotes are known to have evolved from archaea. Even though the domain Archaea cladistically includes eukaryotes, the term archaea in English still generally refers specifically to prokaryotic members of Archaea. Archaea were initially classified as bacteria, receiving the name archaebacteria, but this term has fallen out of use. Archaeal cells have unique properties distinguishing them from Bacteria and Eukaryota, including: cell membranes made of ether-linked lipids; metabolisms such as methanogenesis; and a unique motility structure known as an archaellum. Archaea are further divided into multiple recognized phyla. Classification is difficult because most have not been isolated in a laboratory and have been identified only by their gene sequences in environmental samples. It is unknown if they can produce endospores.
An archipelago, sometimes called an island group or island chain, is a chain, cluster, or collection of islands. An archipelago may be in an ocean, a sea, or a smaller body of water. Examples of archipelagos include the Aegean Islands, the Canadian Arctic Archipelago, the Stockholm Archipelago, the Malay Archipelago, the Lucayan (Bahamian) Archipelago, the Japanese archipelago, and the Hawaiian Archipelago.
The Arctic is the polar region of Earth that surrounds the North Pole, lying north of the Arctic Circle. The Arctic region, from the IERS Reference Meridian travelling east, consists of parts of northern Norway, northernmost Sweden, northern Finland, Russia, the United States (Alaska), Canada, Danish Realm (Greenland), and northern Iceland, along with the Arctic Ocean and adjacent seas.
Armature may refer to:Armature, kinematic chain used in computer animation to simulate the motions of virtual characters
Armature (electrical), one of the two principal electrical components of an electromechanical machine
Armature (sculpture), framework around which a sculpture is built
Armature Studio, video game developer

.
Aromatic compounds or arenes are organic compounds "with a chemistry typified by benzene" and "cyclically conjugated."
The word "aromatic" originates from the past grouping of molecules based on odor, before their general chemical properties were understood. The current definition of aromatic compounds does not have any relation to their odor. Aromatic compounds are now defined as cyclic compounds satisfying Hückel's rule.
Aromatic compounds have the following general properties:Typically unreactive
Often non polar and hydrophobic
High carbon-hydrogen ratio
Burn with a strong sooty yellow flame, due to high C:H ratio
Undergo electrophilic substitution reactions and nucleophilic aromatic substitutions.
Arrhythmias, also known as cardiac arrhythmias, are irregularities in the heartbeat, including when it is too fast or too slow. Essentially, this is anything but normal sinus rhythm. A resting heart rate that is too fast – above 100 beats per minute in adults – is called tachycardia, and a resting heart rate that is too slow – below 60 beats per minute – is called bradycardia. Some types of arrhythmias have no symptoms. Symptoms, when present, may include palpitations or feeling a pause between heartbeats. In more serious cases, there may be lightheadedness, passing out, shortness of breath, chest pain, or decreased level of consciousness. While most cases of arrhythmia are not serious, some predispose a person to complications such as stroke or heart failure. Others may result in sudden death.
Artifact or artefact may refer to:.
An astrolabe is an astronomical instrument dating to ancient times. It serves as a star chart and physical model of the visible half-dome of the sky. Its various functions also make it an elaborate inclinometer and an analog calculation device capable of working out several kinds of problems in astronomy. In its simplest form it is a metal disc with a pattern of wires, cutouts, and perforations that allows a user to calculate astronomical positions precisely. It is able to measure the altitude above the horizon of a celestial body, day or night; it can be used to identify stars or planets, to determine local latitude given local time, to survey, or to triangulate. It was used in classical antiquity, the Byzantine Empire, the Islamic Golden Age, the European Middle Ages and the Age of Discovery for all these purposes.
An atmosphere is a layer of gases that envelop an astronomical object, held in place by the gravity of the object. The name originates from Ancient Greek  ἀτμός (atmós) 'vapour, steam' and  σφαῖρα (sphaîra) 'sphere'. An object acquires most of its atmosphere during its primordial epoch, either by accretion of matter or by outgassing of volatiles. The chemical interaction of the atmosphere with the solid surface can change its fundamental composition, as can photochemical interaction with the Sun. A planet retains an atmosphere for longer durations when the gravity is high and the temperature is low. The solar wind works to strip away a planet's outer atmosphere, although this process is slowed by a magnetosphere. The further a body is from the Sun, the lower the rate of atmospheric stripping.
Atomism is a natural philosophy proposing that the physical universe is composed of fundamental indivisible components known as atoms.
Atrophy is the partial or complete wasting away of a part of the body. Causes of atrophy include mutations, poor nourishment, poor circulation, loss of hormonal support, loss of nerve supply to the target organ, excessive amount of apoptosis of cells, and disuse or lack of exercise or disease intrinsic to the tissue itself. In medical practice, hormonal and nerve inputs that maintain an organ or body part are said to have trophic effects. A diminished muscular trophic condition is designated as atrophy. Atrophy is reduction in size of cell, organ or tissue, after attaining its normal mature growth. In contrast, hypoplasia is the reduction in the cellular numbers of an organ, or tissue that has not attained normal maturity.
An aurora is a natural light display in Earth's sky, predominantly observed in high-latitude regions around the Arctic and Antarctic. The terms northern lights and southern lights are used in the Northern and Southern Hemispheres respectively. Auroras display dynamic patterns of radiant light that appear as curtains, rays, spirals or dynamic flickers covering the entire sky.
Autocracy is a form of government in which absolute power is held by one person, known as an autocrat. It includes both absolute monarchies and dictatorships, while it is contrasted with democracy and other forms of free government. The autocrat has total control over the exercise of civil liberties within the autocracy, choosing under what circumstances they may be exercised, if at all. Governments may also blend elements of autocracy and democracy, forming a mixed type of regime sometimes referred to as anocracy, hybrid regime, or electoral autocracy. The concept of autocracy has been recognized in political philosophy since ancient history.
In developmental psychology and moral, political, bioethical philosophy, autonomy is the capacity to make an informed, uncoerced decision. Autonomous organizations or institutions are independent or self-governing. Autonomy can also be defined from a human resources perspective, where it denotes a level of discretion granted to an employee in their work. In such cases, autonomy is known to generally increase job satisfaction. Self-actualized individuals are thought to operate autonomously of external expectations. In a medical context, respect for a patient's personal autonomy is considered one of many fundamental ethical principles in medicine.
Aviation includes the activities surrounding mechanical flight and the aircraft industry. Aircraft include fixed-wing and rotary-wing types, morphable wings, wing-less lifting bodies, as well as lighter-than-air aircraft such as hot air balloons and airships.
An axon is a long slender projection of a nerve cell or neuron found in most animals that typically conducts electrical impulses known as action potentials away from the nerve cell body. The function of the axon is to transmit information to different neurons, muscles, and glands. In certain sensory neurons, such as those for touch and warmth, the axons are called afferent nerve fibers and the electrical impulse travels along these from the periphery to the cell body and from the cell body to the spinal cord along another branch of the same axon. Axon dysfunction can be the cause of many inherited and many acquired neurological disorders that affect both the peripheral and central neurons. Nerve fibers are classed into three types – group A nerve fibers, group B nerve fibers, and group C nerve fibers. Groups A and B are myelinated, and group C is unmyelinated. These groups include both sensory fibers and motor fibers. Another classification groups only the sensory fibers into four categories: Type I, Type II, Type III, and Type IV.
Bacteria are ubiquitous, mostly free-living organisms often consisting of one biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria were among the first life forms to appear on Earth, and are present in most of its habitats. Bacteria inhabit the air, soil, water, acidic hot springs, radioactive waste, and the deep biosphere of Earth's crust. Bacteria play a vital role in many stages of the nutrient cycle by recycling nutrients and the fixation of nitrogen from the atmosphere. The nutrient cycle includes the decomposition of dead bodies; bacteria are responsible for the putrefaction stage in this process. In the biological communities surrounding hydrothermal vents and cold seeps, extremophile bacteria provide the nutrients needed to sustain life by converting dissolved compounds, such as hydrogen sulphide and methane, to energy. Bacteria also live in mutualistic, commensal and parasitic relationships with plants and animals. Most bacteria have not been characterised and there are many species that cannot be grown in the laboratory. The study of bacteria is known as bacteriology, a branch of microbiology.
Bandwidth commonly refers to:Bandwidth or analog bandwidth, frequency bandwidth, or radio bandwidth, a measure of the width of a frequency range
Bandwidth (computing), the rate of data transfer, bit rate or throughput
Spectral linewidth, the width of an atomic or molecular spectral line.
A barometer is a scientific instrument that is used to measure air pressure. Pressure tendency, which is derived from barometric readings, can forecast short term changes in the weather. Many measurements of air pressure are used within surface weather analysis to help find surface troughs, pressure systems and frontal boundaries.
Basalt is an aphanitic (fine-grained) extrusive igneous rock formed from the rapid cooling of low-viscosity lava rich in magnesium and iron exposed at or very near the surface of a rocky planet or moon. More than 90% of all volcanic rock on Earth is basalt. Rapid-cooling, fine-grained basalt has the same chemical composition and mineralogy as slow-cooling, coarse-grained gabbro. The eruption of basalt lava is observed by geologists at about 20 volcanoes per year. Basalt is also an important rock type on other planetary bodies in the Solar System. For example, the bulk of the plains of Venus, which cover ~80% of the surface, are basaltic; the lunar maria are plains of flood-basaltic lava flows; and basalt is a common rock on the surface of Mars.
Biodiversity is the variability of life on Earth. It can be measured on various levels, for example, genetic variability, species diversity, ecosystem diversity and phylogenetic diversity. Diversity is not distributed evenly on Earth—it is greater in the tropics as a result of the warm climate and high primary productivity in the region near the equator. Tropical forest ecosystems cover less than one-fifth of Earth's terrestrial area and contain about 50% of the world's species. There are latitudinal gradients in species diversity for both marine and terrestrial taxa.
A biome is a distinct geographical region with specific climate, vegetation, animal life, and an ecosystem. It consists of a biological community that has formed in response to its physical environment and regional climate. In 1935, Tansley added the climatic and soil aspects to the idea, calling it ecosystem. The International Biological Program (1964–74) projects popularized the concept of biome.
Biometrics are body measurements and calculations related to human characteristics and features. Biometric authentication is used in computer science as a form of identification and access control. It is also used to identify individuals in groups that are under surveillance.
A biomolecule or biological molecule is loosely defined as a molecule produced by a living organism and essential to one or more typically biological processes. Biomolecules include large macromolecules such as proteins, carbohydrates, lipids, and nucleic acids, as well as small molecules such as vitamins and hormones. A general name for this class of material is biological materials. Biomolecules are an important element of living organisms. They are often endogenous, i.e. produced within the organism, but organisms usually also need exogenous biomolecules, for example certain nutrients, to survive.
The biosphere, also called the ecosphere, is the worldwide sum of all ecosystems. It can also be termed the zone of life on the Earth. The biosphere is virtually a closed system with regard to matter, with minimal inputs and outputs. Regarding energy, it is an open system, with photosynthesis capturing solar energy at a rate of around 100 terawatts. By the most general biophysiological definition, the biosphere is the global ecological system integrating all living beings and their relationships, including their interaction with the elements of the lithosphere, cryosphere, hydrosphere, and atmosphere. The biosphere is postulated to have evolved, beginning with a process of biopoiesis or biogenesis, at least some 3.5 billion years ago.
In telecommunications and computing, bit rate is the number of bits that are conveyed or processed per unit of time.
A blockchain is a distributed ledger with growing lists of records (blocks) that are securely linked together via cryptographic hashes. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Since each block contains information about the previous block, they effectively form a chain, with each additional block linking to the ones before it. Consequently, blockchain transactions are resistant to alteration because, once recorded, the data in any given block cannot be changed retroactively without altering all subsequent blocks and obtaining network consensus to accept these changes.
Botany, also called phytology or plant science, is the branch of natural science and biology that studies plants, especially their anatomy, taxonomy, and ecology. A botanist or plant scientist is a scientist who specialises in this field. "Plant" and "botany" may be defined more narrowly to include only land plants and their study, which is also known as phytology. Phytologists or botanists study approximately 410,000 species of land plants, including some 391,000 species of vascular plants and approximately 20,000 bryophytes.
Boundary or Boundaries may refer to:.
Bureaucracy is a system of organization where laws or regulatory authority are implemented by civil servants. Historically, a bureaucracy was a government administration managed by departments staffed with non-elected officials. Today, bureaucracy is the administrative system governing any large institution, whether publicly owned or privately owned. The public administration in many jurisdictions is an example of bureaucracy, as is any centralized hierarchical structure of an institution, including corporations, societies, nonprofit organizations, and clubs.
In Western musical theory, a cadence is the end of a phrase in which the melody or harmony creates a sense of full or partial resolution, especially in music of the 16th century onwards. A harmonic cadence is a progression of two or more chords that concludes a phrase, section, or piece of music. A rhythmic cadence is a characteristic rhythmic pattern that indicates the end of a phrase. A cadence can be labeled "weak" or "strong" depending on the impression of finality it gives.
Calcification is the accumulation of calcium salts in a body tissue. It normally occurs in the formation of bone, but calcium can be deposited abnormally in soft tissue, causing it to harden. Calcifications may be classified on whether there is mineral balance or not, and the location of the calcification. Calcification may also refer to the processes of normal mineral deposition in biological systems, such as the formation of stromatolites or mollusc shells.
In measurement technology and metrology, calibration is the comparison of measurement values delivered by a device under test with those of a calibration standard of known accuracy. Such a standard could be another measurement device of known accuracy, a device generating the quantity to be measured such as a voltage, a sound tone, or a physical artifact, such as a meter ruler.
The Cambrian is the first geological period of the Paleozoic Era and the Phanerozoic Eon. The Cambrian lasted 51.95 million years from the end of the preceding Ediacaran period 538.8 Ma to the beginning of the Ordovician Period 486.85 Ma.
Capacitance is the ability of an object to store electric charge. It is measured by the change in charge in response to a difference in electric potential, expressed as the ratio of those quantities. Commonly recognized are two closely related notions of capacitance: self capacitance and mutual capacitance. An object that can be electrically charged exhibits self capacitance, for which the electric potential is measured between the object and ground. Mutual capacitance is measured between two components, and is particularly important in the operation of the capacitor, an elementary linear electronic component designed to add capacitance to an electric circuit.
A carcinogen is any agent that promotes the development of cancer. Carcinogens can include synthetic chemicals, naturally occurring substances, physical agents such as ionizing and non-ionizing radiation, and biologic agents such as viruses and bacteria. Most carcinogens act by creating mutations in DNA that disrupt a cell's normal processes for regulating growth, leading to uncontrolled cellular proliferation. This occurs when the cell's DNA repair processes fail to identify DNA damage allowing the defect to be passed down to daughter cells. The damage accumulates over time. This is typically a multi-step process during which the regulatory mechanisms within the cell are gradually dismantled allowing for unchecked cellular division.
Catalysis is the increase in rate of a chemical reaction due to an added substance known as a catalyst. Catalysts are not consumed by the reaction and remain unchanged after the reaction. If the reaction is rapid and the catalyst is recycled quickly, a very small amount of catalyst often suffices; mixing, surface area, and temperature are important factors in reaction rate. Catalysts generally react with one or more reactants to form intermediates that subsequently give the final reaction product, in the process of regenerating the catalyst.
Causality is an influence by which one event, process, state, or subject contributes to the production of another event, process, state, or object where the cause is at least partly responsible for the effect, and the effect is at least partly dependent on the cause. The cause of something may also be described as the reason behind the event or process.
Cavitation in fluid mechanics and engineering normally is the phenomenon in which the static pressure of a liquid reduces to below the liquid's vapor pressure, leading to the formation of small vapor-filled cavities in the liquid. When subjected to higher pressure, these cavities, called "bubbles" or "voids", collapse and can generate shock waves that may damage machinery. As a concrete propeller example: The pressure on the suction side of the propeller blades can be very low and when the pressure falls to that of the vapour pressure of the working liquid, cavities filled with gas vapour can form. The process of the formation of these cavities is referred to as cavitation. If the cavities move into the regions of higher pressure, they will implode or collapse. These shock waves are strong when they are very close to the imploded bubble, but rapidly weaken as they propagate away from the implosion. Cavitation collapse is therefore a significant cause of wear in some engineering contexts. Collapsing voids that implode near to a hard surface cause cyclic stress through repeated implosion. This results in surface fatigue of the material, causing a type of damage also called "cavitation damage" or "cavitation erosion". The most common examples of this kind of wear are to pump impellers, and pipe bends where a sudden change in the direction of fast moving liquid occurs.
Cellulose is an organic compound with the formula (C6H10O5)n, a polysaccharide consisting of a linear chain of several hundred to many thousands of β(1→4) linked D-glucose units. Cellulose is an important structural component of the cell walls of green plants, many forms of algae, and the oomycetes. Some species of bacteria secrete it to form biofilms. Cellulose is the most abundant organic polymer on Earth. The cellulose content of cotton fibre is 90%, that of wood is 40–50%, and that of dried hemp is approximately 57%.
A centrifuge is a device that uses centrifugal force to subject a specimen to a specified constant force – for example, to separate various components of a fluid. This is achieved by spinning the fluid at high speed within a container, thereby separating fluids of different densities or liquids from solids. It works by causing denser substances and particles to move outward in the radial direction. At the same time, objects that are less dense are displaced and moved to the centre. In a laboratory centrifuge that uses sample tubes, the radial acceleration causes denser particles to settle to the bottom of the tube, while low-density substances rise to the top. A centrifuge can be a very effective filter that separates contaminants from the main body of fluid.
A ceramic is any of the various hard, brittle, heat-resistant, and corrosion-resistant materials made by shaping and then firing an inorganic, nonmetallic material, such as clay, at a high temperature. Common examples are earthenware, porcelain, and brick.
Chlorophyll is any of several related green pigments found in cyanobacteria and in the chloroplasts of algae and plants. Its name is derived from the Greek words χλωρός and φύλλον. Chlorophyll allows plants to absorb energy from light. Those pigments are involved in oxygenic photosynthesis, as opposed to bacteriochlorophylls, related molecules found only in bacteria and involved in anoxygenic photosynthesis.
A chromosome is a package of DNA containing part or all of the genetic material of an organism. In most chromosomes, the very long thin DNA fibers are coated with nucleosome-forming packaging proteins; in eukaryotic cells, the most important of these proteins are the histones. Aided by chaperone proteins, the histones bind to and condense the DNA molecule to maintain its integrity. These eukaryotic chromosomes display a complex three-dimensional structure that has a significant role in transcriptional regulation.
In cryptography, a cipher is an algorithm for performing encryption or decryption—a series of well-defined steps that can be followed as a procedure. An alternative, less common term is encipherment. To encipher or encode is to convert information into cipher or code. In common parlance, "cipher" is synonymous with "code", as they are both a set of steps that encrypt a message; however, the concepts are distinct in cryptography, especially classical cryptography.
An electronic circuit is composed of individual electronic components, such as resistors, transistors, capacitors, inductors and diodes, connected by conductive wires or traces through which electric current can flow. It is a type of electrical circuit. For a circuit to be referred to as electronic, rather than electrical, generally at least one active component must be present. The combination of components and wires allows various simple and complex operations to be performed: signals can be amplified, computations can be performed, and data can be moved from one place to another.
Cognitions are mental processes that deal with knowledge. They encompass psychological activities that acquire, store, retrieve, transform, or apply information. Cognitions are a pervasive part of mental life, helping individuals understand and interact with the world.
Coherence is, in general, a state or situation in which all the parts or ideas fit together well so that they form a united whole.
A colloid is a mixture in which one substance, consisting of microscopically dispersed insoluble particles, is suspended throughout another substance. Some definitions specify that the particles must be dispersed in a liquid, while others extend the definition to include substances like aerosols and gels. The term colloidal suspension refers unambiguously to the overall mixture. A colloid has a dispersed phase and a continuous phase.
Combustion, or burning, is a high-temperature exothermic redox chemical reaction between a fuel and an oxidant, usually atmospheric oxygen, that produces oxidized, often gaseous products, in a mixture termed as smoke. Combustion does not always result in fire, because a flame is only visible when substances undergoing combustion vaporize, but when it does, a flame is a characteristic indicator of the reaction. While activation energy must be supplied to initiate combustion, the heat from a flame may provide enough energy to make the reaction self-sustaining. The study of combustion is known as combustion science.
In economics, a commodity is an economic good, usually a resource, that specifically has full or substantial fungibility: that is, the market treats instances of the good as equivalent or nearly so with no regard to who produced them.
A press release is an official statement delivered to members of the news media for the purpose of providing new information, creating an official statement, or making an announcement directed for public release. Press releases are also considered a primary source, meaning they are original informants for information. A press release is traditionally composed of nine structural elements, including a headline, dateline, introduction, body, and other components. Press releases are typically delivered to news media electronically, ready to use, and sometimes subject to "do not use before" time, known as a news embargo.
Conductor or conduction may refer to:.
Conifers are a group of vascular plants and a subset of gymnosperms. They are primarily perennial, woody trees and shrubs, mostly evergreen with a regular branching pattern, reproducing with male and female cones, usually on the same tree. They are wind-pollinated and the seeds are usually dispersed by the wind. Taxonomically, they make up the division Pinophyta, also known as Coniferae. All extant conifers, except for the gnetophytes, are perennial woody plants with secondary growth. There are over 600 living species.
Consensus usually refers to general agreement among a group of people or community. It may also refer to:.
A constellation is an area on the celestial sphere in which a group of visible stars forms a perceived pattern or outline, typically representing an animal, mythological subject, or inanimate object.
Constraint may refer to:Constraint, a demarcation of geometrical characteristics between two or more entities or solid modeling bodies
Constraint (mathematics), a condition of an optimization problem that the solution must satisfy
Constraint (mechanics), a relation between coordinates and momenta
Constraint
Constraint, the degree of statistical dependence between or among variables
Constraints (journal), a scientific journal
Constraint (database), a concept in relational database.
Continuum may refer to:Continuum (measurement), theories or models that explain gradual transitions from one condition to another without abrupt changes.
Convection is single-phase or multiphase fluid flow that occurs spontaneously through the combined effects of material property heterogeneity and body forces on a fluid. When the cause of the convection is unspecified, convection due to the effects of thermal expansion and gravity/buoyancy can be assumed.
Corrosion is a natural process that converts a refined metal into a more chemically stable oxide. It is the gradual deterioration of materials by chemical or electrochemical reaction with their environment. Corrosion engineering is the field dedicated to controlling and preventing corrosion.

where ρ is the density, m is the mass, and V is the volume. In some cases, density is loosely defined as its weight per unit volume, although this is scientifically inaccurate – this quantity is more specifically called specific weight.
Dentition pertains to the development of teeth and their arrangement in the mouth. In particular, it is the characteristic arrangement, type, and number of teeth in a given species at a given age, as well as the morpho-physiology of the animal's teeth.
Deposition may refer to:Deposition (law), taking testimony outside of court
Deposition (politics), the removal of a person of authority from political power
Deposition (university), a widespread initiation ritual for new students practiced from the Middle Ages until the 18th century.
In mathematics, the derivative is a fundamental tool that quantifies the sensitivity to change of a function's output with respect to its input. The derivative of a function of a single variable at a chosen input value, when it exists, is the slope of the tangent line to the graph of the function at that point. The tangent line is the best linear approximation of the function near that input value. The derivative is often described as the instantaneous rate of change, the ratio of the instantaneous change in the dependent variable to that of the independent variable. The process of finding a derivative is called differentiation.
Desalination is the artificial process by which salt water is converted to fresh water.
More generally, desalination is the removal of salts and minerals from a substance.
It is possible to desalinate saltwater, especially sea water, to produce water for human consumption or irrigation, producing brine as a by-product.
Desertification is a type of gradual land degradation of fertile land into arid desert due to a combination of natural processes and human activities.
Determinism is the metaphysical view that all events within the universe can occur only in one possible way. Deterministic theories throughout the history of philosophy have developed from diverse and sometimes overlapping motives and considerations. Like eternalism, determinism focuses on particular events rather than the future as a concept. Determinism is often contrasted with free will, although some philosophers argue that the two are compatible. The antonym of determinism is indeterminism, the view that events are not deterministically caused.
Deviation may refer to:.
A dialect is a variety of language spoken by a particular group of people. This may include dominant and standardized varieties as well as vernacular, unwritten, or non-standardized varieties, such as those used in developing countries or isolated areas. The dialects of the same language are mainly distinguished from each other by differences in linguistic features such as phonology, morphology, syntax and vocabulary.
Diffraction is the deviation of waves from straight-line propagation without any change in their energy due to an obstacle or through an aperture. Diffraction is the same physical effect as interference, but interference is typically applied to superposition of a few waves and the term diffraction is used when many waves are superposed. The term diffraction pattern is used to refer to an image or map of the different directions of the waves after they have been diffracted.
Diffusion is the net movement of anything generally from a region of higher concentration to a region of lower concentration. Diffusion is driven by a gradient in Gibbs free energy or chemical potential. It is possible to diffuse "uphill" from a region of lower concentration to a region of higher concentration, as in spinodal decomposition. Diffusion is a stochastic process due to the inherent randomness of the diffusing entity and can be used to model many real-life stochastic scenarios. Therefore, diffusion and the corresponding mathematical models are used in several fields beyond physics, such as statistics, probability theory, information theory, neural networks, finance, and marketing.
Digestion is the breakdown of large insoluble food compounds into small water-soluble components so that they can be absorbed into the blood plasma. In certain organisms, these smaller substances are absorbed through the small intestine into the blood stream. Digestion is a form of catabolism that is often divided into two processes based on how food is broken down: mechanical and chemical digestion. The term mechanical digestion refers to the physical breakdown of large pieces of food into smaller pieces which can subsequently be accessed by digestive enzymes. Mechanical digestion takes place in the mouth through mastication and in the small intestine through segmentation contractions. In chemical digestion, enzymes break down food into the small compounds that the body can use.
Dinosaurs are a diverse group of reptiles of the clade Dinosauria. They first appeared during the Triassic period, between 243 and 233.23 million years ago (mya), although the exact origin and timing of the evolution of dinosaurs is a subject of active research. They became the dominant terrestrial vertebrates after the Triassic–Jurassic extinction event 201.3 mya and their dominance continued throughout the Jurassic and Cretaceous periods. The fossil record shows that birds are feathered dinosaurs, having evolved from earlier theropods during the Late Jurassic epoch, and are the only dinosaur lineage known to have survived the Cretaceous–Paleogene extinction event approximately 66 mya. Dinosaurs can therefore be divided into avian dinosaurs—birds—and the extinct non-avian dinosaurs, which are all dinosaurs other than birds.
Diplomacy is the communication by representatives of state, intergovernmental, or non-governmental institutions intended to influence events in the international system.
Discourse is a generalization of the notion of a conversation to any form of communication. Discourse is a major topic in social theory, with work spanning fields such as sociology, anthropology, continental philosophy, and discourse analysis. Following work by Michel Foucault, these fields view discourse as a system of thought, knowledge, or communication that constructs our world experience. Since control of discourse amounts to control of how the world is perceived, social theory often studies discourse as a window into power. Within theoretical linguistics, discourse is understood more narrowly as linguistic information exchange and was one of the major motivations for the framework of dynamic semantics. In these expressions, denotations are equated with their ability to update a discourse context.
Displacement may refer to:.
Diversity, diversify, or diverse may refer to:.
Dopamine is a neuromodulatory molecule that plays several important roles in cells. It is an organic chemical of the catecholamine and phenethylamine families. It is an amine synthesized by removing a carboxyl group from a molecule of its precursor chemical, L-DOPA, which is synthesized in the brain and kidneys. Dopamine is also synthesized in plants and most animals. In the brain, dopamine functions as a neurotransmitter—a chemical released by neurons to send signals to other nerve cells. The brain includes several distinct dopamine pathways, one of which plays a major role in the motivational component of reward-motivated behavior. The anticipation of most types of rewards increases the level of dopamine in the brain, and many addictive drugs increase dopamine release or block its reuptake into neurons following release. Other brain dopamine pathways are involved in motor control and in controlling the release of various hormones. These pathways and cell groups form a dopamine system which is neuromodulatory.
A drought is a period of drier-than-normal conditions. A drought can last for days, months or years. Drought often has large impacts on the ecosystems and agriculture of affected regions, and causes harm to the local economy. Annual dry seasons in the tropics significantly increase the chances of a drought developing, with subsequent increased wildfire risks. Heat waves can significantly worsen drought conditions by increasing evapotranspiration. This dries out forests and other vegetation, and increases the amount of fuel for wildfires.
Ductility is the ability of a material to sustain significant plastic deformation before fracture when undergoing tension, i.e. when the relevant elastic modulus is Young's; the equivalent for deforming under bulk compression, i.e. when using the bulk modulus, is malleability. Historically, materials were considered malleable if they were amenable to forming by hammering or rolling. Lead is an example of a material which is substantially more malleable than ductile. Plastic deformation is the permanent distortion of a material under applied stress, as opposed to elastic deformation, which is reversible upon removing the stress. Ductility is a critical mechanical performance indicator, particularly in applications that require materials to bend, stretch, or deform in other ways without breaking. The extent of ductility can be quantitatively assessed using the percent elongation at break, given by the equation:.
Dynamics or dynamic may refer to:.
An ecosystem is a system formed by organisms in interaction with their environment. The biotic and abiotic components are linked together through nutrient cycles and energy flows.
Eddy may refer to:Eddy (surname), surname used by descendants of a number of English, Irish and Scottish families
Eddy, male given name
Eddy, the swirling of a fluid and the reverse current created when the fluid flows past an obstacle
Eddy current, in electromagnetism, loops of electric current induced within conductors by a changing magnetic field.
Eddy current brake, a device used to slow or stop a moving object by generating eddy currents and thus dissipating its kinetic energy as heat.
Eddy (film), a 2015 Italian film
Eddy & The Soul Band, a 1980s Dutch disco group
Eddy, a character on Ed, Edd n Eddy
Eddy covariance, a statistical method used in meteorology
Eddy Test, administered by the US Navy and Marine Corps during and after World War II
Eddy-class tanker, a former British Royal Fleet Auxiliary class.
Egalitarianism is a school of thought within political philosophy that builds on the concept of social equality, prioritizing it for all people. Egalitarian doctrines are generally characterized by the idea that all humans are equal in fundamental worth or moral status. As such, all people should be accorded equal rights and treatment under the law. Egalitarian doctrines have been important in many modern political philosophies and social movements, including the Enlightenment, classical liberalism, libertarianism, feminism, civil rights, and international human rights. Egalitarianism is a major principle of both classical liberalism with its equality of rights, and redistributive left-wing politics with its stress on equality of outcome.
Elasticity often refers to:Elasticity (physics), continuum mechanics of bodies that deform reversibly under stress.
An electrode is an electrical conductor used to make contact with a nonmetallic part of a circuit. In electrochemical cells, electrodes are essential parts that can consist of a variety of materials (chemicals) depending on the type of cell. An electrode may be called either a cathode or anode according to the direction of the electric current, unrelated to the potential difference between electrodes.
In chemistry and manufacturing, electrolysis is a technique that uses direct electric current (DC) to drive an otherwise non-spontaneous biological and physical reaction. Electrolysis is commercially important as a stage in the separation of elements from naturally occurring sources such as ores using an electrolytic cell. The voltage that is needed for electrolysis to occur is called the decomposition potential. The word "lysis" means to separate or break, so in terms, electrolysis would mean "breakdown via electricity.".
An elemental is a mythic supernatural being that is described in occult and alchemical works from around the time of the European Renaissance, and particularly elaborated in the 16th century works of Paracelsus. According to Paracelsus and his subsequent followers, there are four categories of elementals, which are gnomes, undines, sylphs, and salamanders. These correspond to the four Empedoclean elements of antiquity: earth, water, air, and fire, respectively. Terms employed for beings associated with alchemical elements vary by source and gloss.
The elevation of a geographic location is its height above or below a fixed reference point, most commonly a reference geoid, a mathematical model of the Earth's sea level as an equipotential gravitational surface .
The term elevation is mainly used when referring to points on the Earth's surface, while altitude or geopotential height is used for points above the surface, such as an aircraft in flight or a spacecraft in orbit, and depth is used for points below the surface.
Elimination may refer to:.
An emulsion is a mixture of two or more liquids that are normally immiscible owing to liquid-liquid phase separation. Emulsions are part of a more general class of two-phase systems of matter called colloids. Although the terms colloid and emulsion are sometimes used interchangeably, emulsion more narrowly refers to when both phases, dispersed and continuous, are liquids. In an emulsion, one liquid is dispersed in the other. Examples of emulsions include vinaigrettes, homogenized milk, liquid biomolecular condensates, and some cutting fluids for metal working.
The endocrine system is a messenger system in an organism comprising feedback loops of hormones that are released by internal glands directly into the circulatory system and that target and regulate distant organs. In vertebrates, the hypothalamus is the neural control center for all endocrine systems.
An endoskeleton is a structural frame (skeleton) — usually composed of mineralized tissue — on the inside of an animal, overlaid by soft tissues. Endoskeletons serve as structural support against gravity and mechanical loads, and provide anchoring attachment sites for skeletal muscles to transmit force and allow movements and locomotion.
Entropy is a scientific concept, most commonly associated with states of disorder, randomness, or uncertainty. The term and the concept are used in diverse fields, from classical thermodynamics, where it was first recognized, to the microscopic description of nature in statistical physics, and to the principles of information theory. It has found far-ranging applications in chemistry and physics, in biological systems and their relation to life, in cosmology, economics, and information systems including the transmission of information in telecommunication.
An enzyme is a biological macromolecule, usually a protein, that acts as a biological catalyst, accelerating chemical reactions without being consumed in the process. The molecules on which enzymes act are called substrates, which are converted into products. Nearly all metabolic processes within a cell depend on enzyme catalysis to occur at biologically relevant rates. A metabolic pathway is typically composed of a series of enzyme-catalyzed steps. The study of enzymes is known as enzymology, and a related field focuses on pseudoenzymes—proteins that have lost catalytic activity but may retain regulatory or scaffolding functions, often indicated by alterations in their amino acid sequences or unusual 'pseudocatalytic' behavior.
An epidemic is the rapid spread of disease to a large number of hosts in a given population within a short period of time. For example, in meningococcal infections, an attack rate in excess of 15 cases per 100,000 people for two consecutive weeks is considered an epidemic.
Epigenetics is the study of changes in gene expression that occur without altering the DNA sequence. The Greek prefix epi- in epigenetics implies features that are "on top of" or "in addition to" the traditional DNA-sequence-based mechanism of inheritance. Epigenetics usually involves changes that persist through cell division, and affect the regulation of gene expression. Such effects on cellular and physiological traits may result from environmental factors, or be part of normal development.
Equilibrium may refer to:.
Erosion is the action of surface processes that removes soil, rock, or dissolved material from one location on the Earth's crust and then transports it to another location where it is deposited. Erosion is distinct from weathering which involves no movement. Removal of rock or soil as clastic sediment is referred to as physical or mechanical erosion; this contrasts with chemical erosion, where soil or rock material is removed from an area by dissolution. Eroded sediment or solutes may be transported just a few millimetres, or for thousands of kilometres.
An estuary is a partially enclosed coastal body of brackish water where freshwater from rivers or streams meets and mixes with saltwater from the open sea. Estuaries form transition zones between riverine and marine environments and are classified as ecotones, areas where different ecosystems overlap. They are influenced by both marine processes and fluvial processes. The mixing of seawater and freshwater provides high levels of nutrients both in the water column and in sediment, making estuaries among the most productive natural habitats in the world.
Ethnography is a branch of anthropology and the systematic study of individual cultures. It explores cultural phenomena from the point of view of the subject of the study. Ethnography is also a type of social research that involves examining the behavior of the participants in a given social situation and understanding the group members' own interpretation of such behavior.
Etiology is the study of causation or origination. The word is derived from the Greek word αἰτιολογία (aitiología), meaning "giving a reason for". More completely, etiology is the study of the causes, origins, or reasons behind the way that things are, or the way they function, or it can refer to the causes themselves. The word is commonly used in medicine and in philosophy, but also in physics, biology, psychology, political science, geography, cosmology, spatial analysis and theology in reference to the causes or origins of various phenomena.
Evolutionism is a term used to denote the theory of evolution. Its exact meaning has changed over time as the study of evolution has progressed. In the 19th century, it was used to describe the belief that organisms deliberately improved themselves through progressive inherited change (orthogenesis). The teleological belief went on to include cultural evolution and social evolution. In the 1970s, the term "Neo-Evolutionism" was used to describe the idea that "human beings sought to preserve a familiar style of life unless change was forced on them by factors that were beyond their control.".
Excavation may refer to:Archaeological excavation
Excavation (medicine)
Excavation , 2013
Excavation , 2000
Excavation (novel), a 2000 novel by James Rollins
Excavation: A Memoir, a 2014 memoir by Wendy C. Ortiz
Excavation, a 2003 video game by WildTangent.
Excitation, excite, exciting, or excitement may refer to:Excitation (magnetic), provided with an electrical generator or alternator
Exite, a series of racing video games published by Nintendo starting with Excitebike
Excite, web portal owned by IAC
Excite Ballpark, located in San Jose, California
Electron excitation, the transfer of an electron to a higher atomic orbital
More generally, the transfer of energy to a normal mode
Excitement (film), a lost 1924 silent comedy by Robert F. Hill
Sexual excitation
Stimulation or excitation or excitement, the action of various agents on nerves, muscles, or a sensory end organ, by which activity is evoked
"Exciting", a song by Hieroglyphics from the album The Kitchen.
An exoplanet or extrasolar planet is a planet outside of the Solar System. The first confirmed detection of an exoplanet was in 1992 around a pulsar, and the first detection around a main-sequence star was in 1995. A different planet, first detected in 1988, was confirmed in 2003. In 2016, it was recognized that the first possible evidence of an exoplanet had been noted in 1917. As of 26 February 2026, there are 6,128 confirmed exoplanets in 4,560 planetary systems, with 1,038 systems having more than one planet.
Extinction is the termination of a species via the death of its last member. A taxon may become functionally extinct before the death of its last member if it loses the capacity to reproduce and recover. As a species' potential range may be very large, determining this moment is difficult, and is usually done retrospectively. This difficulty leads to phenomena such as Lazarus taxa, where a species presumed extinct abruptly "reappears" after a period of apparent absence.
A famine is a widespread scarcity of food caused by several possible factors, including, but not limited to: war, natural disasters, crop failure, widespread poverty, an economic catastrophe or government policies. This phenomenon is usually accompanied or followed by regional malnutrition, starvation, epidemic, and increased mortality. Every inhabited continent in the world has experienced a period of famine throughout history. During the 19th and 20th centuries, Southeast and South Asia, as well as Eastern and Central Europe, suffered the greatest number of fatalities due to famine. Deaths caused by famine declined sharply beginning in the 1970s, with numbers falling further since 2000. Since 2010, Africa has been the most affected continent in the world by famine. As of 2025, Haiti and Afghanistan are the two countries with the most catastrophic and widespread states of famine, followed by Sudan.
A fandom is a subculture composed of fans characterized by a feeling of camaraderie with others who share a common interest. Fans typically are interested in even minor details of the objects of their fandom and spend a significant portion of their time and energy involved with their interest, often as a part of a social network with particular practices, differentiating fandom-affiliated people from those with only a casual interest.
Fertility in colloquial terms refers the ability to have offspring. In demographic contexts, fertility refers to the actual production of offspring, rather than the physical capability to reproduce, which is termed fecundity. The fertility rate is the average number of children born during an individual's lifetime. In medicine, fertility refers to the ability to have children, and infertility refers to difficulty in reproducing naturally. In general, infertility or subfertility in humans is defined as not being able to conceive a child after one year of unprotected sex. The antithesis of fertility is infertility, while the antithesis of fecundity is sterility.
Fiction is any creative work, chiefly any narrative work, portraying individuals, events, or places that are imaginary or in ways that are imaginary. Fictional portrayals are thus inconsistent with fact, history, or plausibility. In a traditional narrow sense, fiction refers to written narratives in prose – often specifically novels, novellas, and short stories. More broadly, however, fiction encompasses imaginary narratives expressed in any medium, including not just writings but also live theatrical performances, films, television programs, radio dramas, comics, role-playing games, and video games.
Filtration is a physical separation process that separates solid matter and fluid from a mixture using a filter medium that has a complex structure through which only the fluid can pass. Solid particles that cannot pass through the filter medium are described as oversize and the fluid that passes through is called the filtrate. Oversize particles may form a filter cake on top of the filter and may also block the filter lattice, preventing the fluid phase from crossing the filter, known as blinding. The size of the largest particles that can successfully pass through a filter is called the effective pore size of that filter. The separation of solid and fluid is imperfect; solids will be contaminated with some fluid and filtrate will contain fine particles. Filtration occurs both in nature and in engineered systems; there are biological, geological, and industrial forms. In everyday usage the verb "strain" is more often used; for example, using a colander to drain cooking water from cooked pasta.
Fission, a splitting of something into two or more parts, may refer to:.
Flora is all the plant life present in a particular region or time, generally the naturally occurring (indigenous) native plants. The corresponding term for animals is fauna, and for fungi, it is funga. Sometimes bacteria and fungi are also referred to as flora as in the terms gut flora or skin flora for purposes of specificity.
Flotation involves phenomena related to the relative buoyancy of objects.
Flux describes any effect that appears to pass or travel through a surface or substance. Flux is a concept in applied mathematics and vector calculus which has many applications in physics. For transport phenomena, flux is a vector quantity, describing the magnitude and direction of the flow of a substance or property. In vector calculus, flux is a scalar quantity, defined as the surface integral of the perpendicular component of a vector field over a surface.
A leaf is a principal appendage of the stem of a vascular plant, usually borne laterally above ground and specialized for photosynthesis. Leaves are collectively called foliage, as in "autumn foliage", while the leaves, stem, flower, and fruit collectively form the shoot system. In most leaves, the primary photosynthetic tissue is the palisade mesophyll and is located on the upper side of the blade or lamina of the leaf, but in some species, including the mature foliage of Eucalyptus, palisade mesophyll is present on both sides and the leaves are said to be isobilateral. The leaf is an integral part of the stem system, and most leaves are flattened and have distinct upper (adaxial) and lower (abaxial) surfaces that differ in color, hairiness, the number of stomata, the amount and structure of epicuticular wax, and other features. Leaves are mostly green in color due to the presence of a compound called chlorophyll, which is essential for photosynthesis as it absorbs light energy from the Sun. A leaf with lighter-colored or white patches or edges is called a variegated leaf.
Forensic science, often confused with criminalistics, is the application of science principles and methods to support decision-making related to rules or law, generally criminal and civil law.
In speech science and phonetics, a formant is the broad spectral maximum that results from an acoustic resonance of the human vocal tract. In acoustics, a formant is usually defined as a broad peak, or local maximum, in the spectrum. For harmonic sounds, with this definition, the formant frequency is sometimes taken as that of the harmonic that is most augmented by a resonance. The difference between these two definitions resides in whether "formants" characterise the production mechanisms of a sound or the produced sound itself. In practice, the frequency of a spectral peak differs slightly from the associated resonance frequency, except when, by luck, harmonics are aligned with the resonance frequency, or when the sound source is mostly non-harmonic, as in whispering and vocal fry.
Formation may refer to:.
Fracture is the appearance of a crack or complete separation of an object or material into two or more pieces under the action of stress. The fracture of a solid usually occurs due to the development of certain displacement discontinuity surfaces within the solid. If a displacement develops perpendicular to the surface, it is called a normal tensile crack or simply a crack; if a displacement develops tangentially, it is called a shear crack, slip band, or dislocation.
Friction is the force resisting the relative motion of solid surfaces, fluid layers, and material elements sliding or grinding against each other. Types of friction include dry, fluid, lubricated, skin, and internal – an incomplete list. The study of the processes involved is called tribology, and has a history of more than 2,000 years.
Frost is a layer of ice on a solid surface, which forms from water vapor that deposits onto a freezing surface. Frost forms when the air contains more water vapor than it can normally hold at a specific temperature. The process is similar to the formation of dew, except it occurs below the freezing point of water typically without crossing through a liquid state.
In classical music, a fugue is a contrapuntal, polyphonic compositional technique in two or more voices, built on a subject that is introduced at the beginning in imitation, which recurs frequently throughout the course of the composition. It is not to be confused with a fuguing tune, which is a style of song popularized by and mostly limited to early American music and West Gallery music. A fugue usually has three main sections: an exposition, a development, and a final entry that contains the return of the subject in the fugue's tonic key. Fugues can also have episodes, which are parts of the fugue where new material often based on the subject is heard; a stretto, when the fugue's subject overlaps itself in different voices, or a recapitulation. A popular compositional technique in the Baroque era, the fugue was fundamental in showing mastery of harmony and tonality as it presented counterpoint.
A fungus is any member of the group of eukaryotic organisms that includes microorganisms such as yeasts and molds, as well as the more familiar mushrooms. These organisms are classified in the kingdom Fungi.
Fusion, or synthesis, is the process of combining two or more distinct entities into a new whole.
A galaxy is a system of stars, stellar remnants, interstellar gas, dust, and dark matter bound together by gravity. The word is derived from the Greek galaxias (γαλαξίας), meaning 'milky', a reference to the Milky Way galaxy that contains the Solar System. Galaxies, averaging an estimated 100 million stars, range in size from dwarfs with less than a thousand stars to the largest galaxies known—supergiants with one hundred trillion stars, each orbiting its galaxy's center of mass. Most of the mass in a typical galaxy is in the form of dark matter, with only a few percent of that mass visible in the form of stars and nebulae. Supermassive black holes are a common feature at the centers of galaxies.

Gallium is a chemical element; it has symbol Ga and atomic number 31. Discovered by the French chemist Paul-Émile Lecoq de Boisbaudran in Paris, France, 1875,
elemental gallium is a soft, silvery metal at standard temperature and pressure. In its liquid state, it becomes silvery white. If enough force is applied, solid gallium may fracture conchoidally. Since its discovery in 1875, gallium has widely been used to make alloys with low melting points. It is also used in semiconductors, as a dopant in semiconductor substrates.
Gametogenesis is a biological process by which diploid or haploid precursor cells undergo cell division and differentiation to form mature haploid gametes. Depending on the biological life cycle of the organism, gametogenesis occurs by meiotic division of diploid gametocytes into various gametes, or by mitosis. For example, plants produce gametes through mitosis in gametophytes. The gametophytes grow from haploid spores after sporic meiosis. The existence of a multicellular, haploid phase in the life cycle between meiosis and gametogenesis is also referred to as alternation of generations.
A gantry is an overhead bridge-like structure supporting equipment such as a crane, signals, or cameras.
Gasification is a process that converts biomass- or fossil fuel-based carbonaceous materials into gases, including as the largest fractions: dinitrogen (N2), carbon monoxide (CO), dihydrogen (H2), and carbon dioxide (CO2). This is achieved by reacting the feedstock material at high temperatures (typically >700 °C), without combustion, via controlling the amount of oxygen and/or steam present in the reaction. The resulting gas mixture is called syngas (from synthesis gas) or producer gas and is itself a fuel due to the flammability of the H2 and CO of which the gas is largely composed. Power can be derived from the subsequent combustion of the resultant gas, and is considered to be a source of renewable energy if the gasified compounds were obtained from biomass feedstock.
Gastronomy is the study of the relationship between food and culture, the art of preparing and serving rich or delicate and appetizing food, the cooking styles of particular regions, and the science of good eating. One who is well versed in gastronomy is called a gastronome, while a gastronomist is one who unites theory and practice in the study of gastronomy. Practical gastronomy is associated with the practice and study of the preparation, production, and service of the various foods and beverages, from countries around the world. It is related with a system and process approach, focused on recipes, techniques and cookery books. Food gastronomy is connected with food and beverages and their genesis. Technical gastronomy underpins practical gastronomy, introducing a rigorous approach to evaluation of gastronomic topics.
Genealogy is the study of families, family history, and the tracing of their lineages. Genealogists use oral interviews, historical records, genetic analysis, and other records to obtain information about a family and to demonstrate kinship and pedigrees of its members. The results are often displayed in charts or written as narratives. The field of family history is broader than genealogy, and covers not just lineage but also family and community history and biography.
Geodesy or geodetics is the science of measuring and representing the geometry, gravity, and spatial orientation of the Earth in temporally varying 3D space. It is called planetary geodesy when studying other astronomical bodies, such as planets or circumplanetary systems. Geodetic job titles include geodesist and geodetic surveyor.
A geoglyph is a large design or motif – generally longer than 4 metres (13 ft) – produced on the ground by durable elements of the landscape, such as stones, stone fragments, gravel, or earth. A positive geoglyph is formed by the arrangement and alignment of materials on the ground in a manner akin to petroforms, while a negative geoglyph is formed by removing part of the natural ground surface to create differently coloured or textured ground in a manner akin to petroglyphs.
Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.".
Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science.
Geometry is a branch of mathematics concerned with properties of space such as the distance, shape, size, and relative position of figures. Geometry is, along with arithmetic, one of the oldest branches of mathematics. A mathematician who works in the field of geometry is called a geometer. Until the 19th century, geometry was almost exclusively devoted to Euclidean geometry, which includes the notions of point, line, plane, distance, angle, surface, and curve, as fundamental concepts.
Geophysics is a physical science concerned with the processes and properties of Earth and its surrounding space environment, studied using quantitative and observational methods. It focuses primarily on Earth’s shape and its gravitational, magnetic, and electromagnetic fields. It also studies internal structure, composition, and dynamics, and their surface expression in tectonics, volcanism, and rock formation. Geophysics also encompasses a broader Earth-system and planetary perspective, including the oceans, atmosphere, cryosphere, ionosphere, magnetosphere, as well as solar–terrestrial interactions and analogous processes on the Moon, other planets, and their satellites.
Germination is the process by which an organism grows from a seed or spore. The term is applied to the sprouting of a seedling from a seed of an angiosperm or gymnosperm, the growth of a sporeling from a spore, such as the spores of fungi, ferns, bacteria, and the growth of the pollen tube from the pollen grain of a seed plant.
A geyser is a spring with an intermittent water discharge ejected turbulently and accompanied by steam. The formation of geysers is fairly rare and is caused by particular hydrogeological conditions that exist only in a few places on Earth.
A glacier is a persistent body of dense ice, a form of rock, that is constantly moving downhill under its own weight. A glacier forms where the accumulation of snow exceeds its ablation over many years, often centuries. It acquires distinguishing features, such as crevasses and seracs, as it slowly flows and deforms under stresses induced by its weight. As it moves, it abrades rock and debris from its substrate to create landforms such as cirques, moraines, or fjords. Although a glacier may flow into a body of water, it forms only on land and is distinct from the much thinner sea ice and lake ice that form on the surface of bodies of water.
Globalism has multiple meanings. In political science, it is used to describe "attempts to understand all of the interconnections of the modern world—and to highlight patterns that underlie them". While primarily associated with world-systems, it can be used to describe other global trends. The concept of globalism is also classically used to focus on ideologies of globalisation instead of its processes ; in this sense, "globalism" is to globalisation what "nationalism" is to nationalisation.
A glyph is any kind of purposeful mark. In typography, a glyph is "the specific shape, design, or representation of a character". It is a particular graphical representation, in a particular typeface, of an element of written language.
Granite is a coarse-grained (phaneritic) intrusive igneous rock composed mostly of quartz, alkali feldspar, mica and plagioclase. It forms from magma with a high content of silica and alkali metal oxides that slowly cools and solidifies underground. It is common in the continental crust of Earth, where it is found in igneous intrusions. These range in size from dikes only a few centimeters across to batholiths exposed over hundreds of square kilometers.
In physics, gravity, also known as gravitation or a gravitational interaction, is a fundamental interaction, which may be described as the force that draws material objects towards each other.
A greenhouse is a structure that is designed to regulate the temperature and humidity of the environment inside. There are different types of greenhouses, but they all have large areas covered with transparent materials that let sunlight pass and block it as heat. The most common materials used in modern greenhouses for walls and roofs are rigid plastic made of polycarbonate, plastic film made of polyethylene, or glass panes. When the inside of a greenhouse is exposed to sunlight, the temperature increases, providing a sheltered environment for plants to grow even in cold weather.
In ecology, habitat refers to the array of resources, biotic factors that are present in an area, such as to support the survival and reproduction of a particular species. A species' habitat can be seen as the physical manifestation of its ecological niche. Thus "habitat" is a species-specific term, fundamentally different from concepts such as environment or vegetation assemblages, for which the term "habitat-type" is more appropriate.
The halogens are a group in the periodic table consisting of six chemically related elements: fluorine (F), chlorine (Cl), bromine (Br), iodine (I), and the radioactive elements astatine (At) and tennessine (Ts), though some authors would exclude tennessine as its chemistry is unknown and is theoretically expected to be more like that of gallium. In the modern IUPAC nomenclature, this group is known as group 17.
In music, harmony is the concept of combining different sounds in order to create new, distinct musical ideas. Theories of harmony seek to describe or explain the effects created by distinct pitches or tones coinciding with one another; harmonic objects such as chords, textures and tonalities are identified, defined, and categorized in the development of these theories. Harmony is broadly understood to involve both a "vertical" dimension (frequency-space) and a "horizontal" dimension (time-space), and often overlaps with related musical concepts such as melody, timbre, and form.
Harvesting is the process of collecting plants, animals, or fish as food, especially the process of gathering mature crops, and "the harvest" also refers to the collected crops. Reaping is the cutting of grain or pulses for harvest, typically using a scythe, sickle, or reaper. On smaller farms with minimal mechanization, harvesting is the most labor-intensive activity of the growing season. On large mechanized farms, harvesting uses farm machinery, such as the combine harvester. Automation has increased the efficiency of both the seeding and harvesting processes. Specialized harvesting equipment, using conveyor belts for gentle gripping and mass transport, replaces the manual task of removing each seedling by hand. The term "harvesting" in general usage may include immediate postharvest handling, including cleaning, sorting, packing, and cooling.
Hematology is the branch of medicine concerned with the study of the cause, prognosis, treatment, and prevention of diseases related to blood. It involves treating diseases that affect the production of blood and its components, such as blood cells, hemoglobin, blood proteins, bone marrow, platelets, blood vessels, spleen, and the mechanism of coagulation. Such diseases might include hemophilia, sickle cell anemia, blood clots (thrombus), other bleeding disorders, and blood cancers such as leukemia, multiple myeloma, and lymphoma. The laboratory analysis of blood is frequently performed by a medical technologist or medical laboratory scientist.
A herbivore is an animal anatomically and physiologically evolved to feed on plants, especially upon vascular tissues such as foliage, fruits or seeds, as the main component of its diet. These more broadly also encompass animals that eat non-vascular autotrophs such as mosses, algae and lichens, but do not include those feeding on decomposed plant matters or macrofungi.
A hierarchy is an arrangement of items that are represented as being "above", "below", or "at the same level as" one another. Hierarchy is an important concept in a wide variety of fields, such as architecture, philosophy, design, mathematics, computer science, organizational theory, systems theory, systematic biology, and the social sciences.
Holography is a technique that allows a wavefront to be recorded and later reconstructed. It is best known as a method of generating three-dimensional images, and has a wide range of other uses, including data storage, microscopy, and interferometry. In principle, it is possible to make a hologram for any type of wave.
The Hominidae, whose members are known as the great apes, are a taxonomic family of primates that includes eight extant species in four genera: Pongo ; Gorilla ; Pan ; and Homo, of which only modern humans  remain.
Humidity is the concentration of water vapor present in the air. Water vapor, the gaseous state of water, is generally invisible to the naked eye. Humidity indicates the likelihood for precipitation, dew, or fog to be present.
Hybridization may refer to:Hybridization (biology), the process of combining different varieties of organisms to create a hybrid
Orbital hybridization, in chemistry, the mixing of atomic orbitals into new hybrid orbitals
Nucleic acid hybridization, the process of joining two complementary strands of nucleic acids - RNA, DNA or oligonucleotides
In evolutionary algorithms, the merging two or more optimization techniques into a single algorithm
Memetic algorithm, a common template for hybridization
In linguistics, the process of one variety blending with another variety
The alteration of a vehicle into a hybrid electric vehicle
In globalization theory, the ongoing blending of cultures
Hybridization in political election campaign communication, the combining of campaign techniques developed in different countries
In paleoanthropology, the hypothesis of Neanderthal and human hybridization.
Hydration may refer to:Hydrate, a substance that contains water
Hydration enthalpy, energy released through hydrating a substance
Hydration reaction, a chemical addition reaction where a hydroxyl group and proton are added to a compound
Hydration shell, a type of solvation shell
Hydration system, an apparatus that helps its user drink enough liquid while engaged in physical activity
Hydration pack, a type of hydration system composed of a carry-on pack used for hydration
Mineral hydration, an inorganic chemical reaction where water is added to the crystal structure of a mineral
Drinking in general, including:
Fluid replacement, the medical practice of replenishing bodily fluid
Oral rehydration therapy, hydration as a health treatment
Management of dehydration, medical hydration
Tissue hydration, the supply and retention of adequate water in biological tissues
Water of hydration, water that occurs within crystals
Hydration, the preparation of web page content for user interaction.
Dough hydration, the percentage of water in a dough in relation to the amount of flour.
Hydraulics is a technology and applied science using engineering, chemistry, and other sciences involving the mechanical properties and use of liquids.
In organic chemistry, a hydrocarbon is an organic compound consisting entirely of hydrogen and carbon. Hydrocarbons are examples of group 14 hydrides. Hydrocarbons are generally colourless and hydrophobic; their odor is usually faint, and may be similar to that of gasoline or lighter fluid. They occur in a diverse range of molecular structures and phases: they can be gases, liquids, low melting solids or polymers.
Hydrology is the scientific study of the movement, distribution, and management of water on Earth and other planets, including the water cycle, water resources, and drainage basin sustainability. A practitioner of hydrology is called a hydrologist. Hydrologists are scientists studying earth or environmental science, civil or environmental engineering, and physical geography. Using various analytical methods and scientific techniques, they collect and analyze data to help solve water related problems such as environmental preservation, natural disasters, and water management.
A hypothesis is a proposed explanation for a phenomenon. A scientific hypothesis must be based on observations and make a testable and reproducible prediction about reality, in a process beginning with an educated guess or thought.
If a hypothesis is repeatedly independently demonstrated by experiment to be true, it becomes a scientific theory. In colloquial usage, the words hypothesis and theory are often used interchangeably, but this is incorrect in the context of science.
An iceberg is a piece of fresh water ice more than 15 meters long that has broken off a glacier or an ice shelf and is floating freely in open water. Smaller chunks of floating glacially derived ice are called "growlers" or "bergy bits". Much of an iceberg is below the water's surface, which led to the expression "tip of the iceberg" to illustrate a small part of a larger unseen issue. Icebergs are considered a serious maritime hazard.
Iconography, as a branch of art history, studies the identification, description and interpretation of the content of images: the subjects depicted, the particular compositions and details used to do so, and other elements that are distinct from artistic style. The word iconography comes from the Greek εἰκών ("image") and γράφειν.
An ideology is a set of beliefs or values attributed to a person or group of persons, especially those held for reasons that are not purely about belief in certain knowledge, in which "practical elements are as prominent as theoretical ones". Formerly applied primarily to economic, political, or religious theories and policies, in a tradition going back to Karl Marx and Friedrich Engels, more recent use treats the term as mainly condemnatory.
Ignition may refer to:.
Illumination may refer to:.
Imaging is the process of creating visual representations of objects, scenes, or phenomena. The term encompasses both the formation of images through physical processes and the technologies used to capture, store, process, and display them. While traditional imaging relies on visible light, modern imaging systems can visualize information across the electromagnetic spectrum and through other physical phenomena such as sound waves, magnetic fields, and particle emissions, enabling the visualization of subjects invisible to the human eye.
Immigration is the international movement of people to a destination country of which they are not usual residents or where they do not possess nationality in order to settle as permanent residents. Commuters, tourists, and other short-term stays in a destination country do not fall under the definition of immigration or migration; seasonal labour immigration is sometimes included, however.
Immunity may refer to:.
Implosion can refer to:.
The word incubation may refer to:.
Index may refer to:.
Induction or inductive may refer to:.
Inequality may refer to:Inequality (mathematics), a relation between two quantities when they are different.
Economic inequality, difference in economic well-being between population groups
Income inequality, an unequal distribution of income
Wealth inequality, an unequal distribution of wealth
Spatial inequality, the unequal distribution of income and resources across geographical regions
International inequality, economic differences between countries
Social inequality, unequal opportunities and rewards for different social positions or statuses within a group
Gender inequality, unequal treatment or perceptions due to gender
Racial inequality, social distinctions between racial and ethnic groups within a society
Health inequality, differences in the quality of health and healthcare across populations
Educational inequality, the unequal distribution of academic resources
Environmental inequality, unequal environmental harms between different neighborhoods or cities
Urban forest inequity, an unequal distribution of trees
Attention inequality, unequal distribution of attention across users, groups of people, issues in etc. in attention economy
Participation inequality, the phenomenon in which a small percentage of people contributes the majority of information to the total outcome.
Inertia is the natural tendency of objects in motion to stay in motion and objects at rest to stay at rest, unless a force causes its velocity to change. It is one of the fundamental principles in classical physics, and described by Isaac Newton in his first law of motion. It is one of the primary manifestations of mass, one of the core quantitative properties of physical systems. Newton writes:LAW I. Every object perseveres in its state of rest, or of uniform motion in a right line, except insofar as it is compelled to change that state by forces impressed thereon.
Inferences are steps in logical reasoning, moving from premises to logical consequences; etymologically, the word infer means to "carry forward". Inference is theoretically traditionally divided into deduction and induction, a distinction that dates at least to Aristotle. Deduction is inference deriving logical conclusions from premises known or assumed to be true, with the laws of valid inference being studied in logic. Induction is inference from particular evidence to a universal conclusion. A third type of inference is sometimes distinguished, notably by Charles Sanders Peirce, contradistinguishing abduction from induction.
In economics, inflation is an increase in the average price of goods and services in terms of money. This increase is measured using a price index, typically a consumer price index (CPI). When the general price level rises, each unit of currency buys fewer goods and services; consequently, inflation corresponds to a reduction in the purchasing power of money. The opposite of inflation is deflation, a decrease in the general price level of goods and services. The common measure of inflation is the inflation rate, the annualized percentage change in a general price index.
Infrared is electromagnetic radiation (EMR) with wavelengths longer than that of visible light but shorter than microwaves. The infrared spectral band begins with the waves that are just longer than those of red light, so IR is invisible to the human eye. IR is generally understood to include wavelengths from around 780 nm (380 THz) to 1 mm (300 GHz). IR is commonly divided between longer-wavelength thermal IR, emitted from terrestrial sources, and shorter-wavelength IR, or near IR, part of the solar spectrum. Longer IR wavelengths (30–100 μm) are sometimes included as part of the terahertz radiation band. Almost all black-body radiation from objects near room temperature is in the IR band. As a form of EMR, IR carries energy and momentum, exerts radiation pressure, and has properties corresponding to both those of a wave and of a particle, the photon.
In law and conflict of laws, domicile is relevant to an individual's "personal law", which includes the law that governs a person's status and their property. It is independent of a person's nationality. Although a domicile may change from time to time, a person has only one domicile at any point in their life, no matter what their circumstances. Domicile is distinct from habitual residence, where there is less focus on future intent.
Injection or injected may refer to:.
Innovation is the practical implementation of ideas that result in the introduction of new goods or services or improvement in offering goods or services. ISO TC 279 in the standard ISO 56000:2020 defines innovation as "a new or changed entity, realizing or redistributing value". Others have different definitions; a common element in the definitions is a focus on newness, improvement, and spread of ideas or technologies.
Insects are hexapod invertebrates of the class Insecta. They are the largest group within the arthropod phylum. Insects have a chitinous exoskeleton, a three-part body, three pairs of jointed legs, compound eyes, and a pair of antennae. Insects are the most diverse group of animals, with more than a million described species; they represent more than half of all animal species.
Insertion may refer to:Insertion (anatomy), the point of a tendon or ligament onto the skeleton or other part of the body
Insertion (genetics), the addition of DNA into a genetic sequence
Insertion, several meanings in medicine, see ICD-10-PCS
Insertion loss, in electronics
Insertion reaction, a chemical reaction in which one chemical entity interposes itself into an existing bond of a second chemical entity
Insertion sort, a simple computer algorithm for sorting arrays
Local insertion, in broadcasting
Insertion of a character in a string, one of the single-character edits used to define the Levenshtein distance.
Insomnia, also known as sleeplessness, is a sleep disorder causing difficulty falling asleep or staying asleep for as long as desired. Insomnia is typically followed by daytime sleepiness, low energy, irritability, and a depressed mood. It may result in an increased risk of accidents as well as problems focusing and learning. Insomnia can be short-term, lasting for days or weeks, or long-term, lasting more than a month. The concept of the word insomnia has two distinct possibilities: insomnia disorder or insomnia symptoms.
An inspection is, most generally, an organized examination or formal evaluation exercise. In engineering activities inspection involves the measurements, tests, and gauges applied to certain characteristics in regard to an object or activity. The results are usually compared to specified requirements and standards for determining whether the item or activity is in line with these targets, often with a Standard Inspection Procedure in place to ensure consistent checking. Inspections are usually non-destructive.
In dynamical systems instability means that some of the outputs or internal states increase with time, without bounds. Not all systems that are not stable are unstable; systems can also be marginally stable or exhibit limit cycle behavior.
Instrumentation is a collective term for measuring instruments, used for indicating, measuring, and recording physical quantities. It is also a field of study about the art and science about making measurement instruments, involving the related areas of metrology, automation, and control theory. The term has its origins in the art and science of scientific instrument-making.
Insulation may refer to:.
Intensity may refer to:.
Interaction is action that occurs between two or more entities, generally used in philosophy and the sciences. It may refer to:.
Interface or interfacing may refer to:.
Interference is the act of interfering, invading, or poaching. Interference may also refer to:.
The intertidal zone or foreshore is the area above water level at low tide and underwater at high tide; in other words, it is the part of the littoral zone within the tidal range. This area can include several types of habitats with various species of life, such as sea stars, sea urchins, and many species of coral with regional differences in biodiversity. Sometimes it is referred to as the littoral zone or seashore, although those can be defined as a wider region.
Inversion or inversions may refer to:.
Ionization or ionisation is the process by which an atom or a molecule acquires a negative or positive charge by gaining or losing electrons, often in conjunction with other chemical changes. The resulting electrically charged atom or molecule is called an ion. Ionization can result from the loss of an electron after collisions with subatomic particles, collisions with other atoms, molecules, electrons, positrons, protons, antiprotons, and ions, or through the interaction with electromagnetic radiation. Heterolytic bond cleavage and heterolytic substitution reactions can result in the formation of ion pairs. Ionization can occur through radioactive decay by the internal conversion process, in which an excited nucleus transfers its energy to one of the inner-shell electrons causing it to be ejected.

Irradiation is the process by which an object is exposed to radiation. An irradiator is a device used to expose an object to radiation, most often gamma radiation, for a variety of purposes. Irradiators may be used for sterilizing medical and pharmaceutical supplies, preserving foodstuffs, alteration of gemstone colors, studying radiation effects, eradicating insects through sterile male release programs, or calibrating thermoluminescent dosimeters (TLDs).
Irrigation is the practice of applying controlled amounts of water to land to help grow crops, landscape plants, and lawns. Irrigation has been a key aspect of agriculture for over 5,000 years and has been developed by many cultures around the world. Irrigation helps to grow crops, maintain landscapes, and revegetate disturbed soils in dry areas and during times of below-average rainfall. In addition to these uses, irrigation is also employed to protect crops from frost, suppress weed growth in grain fields, and prevent soil consolidation. It is also used to cool livestock, reduce dust, dispose of sewage, and support mining operations. Drainage, which involves the removal of surface and sub-surface water from a given location, is often studied in conjunction with irrigation.
Isobar may refer to:Isobar (meteorology), a line on a map or chart connecting points of equal atmospheric pressure reduced to sea level.
Isobaric process, a process taking place at constant pressure
Isobar (nuclide), one of multiple nuclides with the same mass but with different numbers of protons.
Isotopes are distinct nuclear species of the same chemical element. They have the same atomic number and position in the periodic table, but different nucleon numbers due to different numbers of neutrons in their nuclei. While all isotopes of a given element have virtually the same chemical properties, they have different atomic masses and physical properties.
Iteration means repeating a process to generate a sequence of outcomes. Each repetition of the process is a single iteration, and the outcome of each iteration is the starting point of the next iteration.
Jet streams are fast flowing, narrow air currents in the atmosphere. The main terrestrial jet streams are located near the altitude of the tropopause and are westerly winds, flowing west to east around the globe. The Northern Hemisphere and the Southern Hemisphere each have a polar jet around their respective polar vortex at around 30,000 ft above sea level and typically travelling at around 110 mph (180 km/h) although often considerably faster. Closer to the equator, somewhat higher and somewhat weaker, is a subtropical jet.
Jurisdiction is the legal term for the legal authority held by a legal entity to enact justice. Jurisdiction is rarely claimed to be complete: rather it is limited for example by geography, subject matter, or other factor. It is only within the scope of such jurisdiction that, for example, the parties to a dispute have standing to bring the matter before a judge, who has power to decide it authoritatively.
A karyotype is the general appearance of the complete set of chromosomes in the cells of a species or in an individual organism, mainly including their sizes, numbers, and shapes. Karyotyping is the process by which a karyotype is discerned by determining the chromosome complement of an individual, including the number of chromosomes and any abnormalities.
Kinetics may refer to:.
In Greek mythology, the Labyrinth is an elaborate, confusing structure designed and built by the mythological artificer Daedalus for King Minos of Crete at Knossos. Its function was to hold the Minotaur, the monster eventually killed by the hero Theseus. Daedalus had so cunningly made the Labyrinth that he could barely escape it after he built it.
Lactation describes the secretion of milk from the mammary glands in addition to the period of time that a parent lactates to feed her young. The process can occur with all sexually mature female mammals, although it may predate mammals. The process of feeding milk in all female creatures is called nursing, and in humans it is also called breastfeeding. Newborn infants often produce some milk from their own breast tissue, known colloquially as witch's milk.
Lamination is the technique/process of manufacturing a material in multiple layers, so that the composite material achieves improved strength, stability, sound insulation, appearance, or other properties from the use of the differing materials, such as plastic. A laminate is a layered object or material assembled using heat, pressure, welding, or adhesives. Various coating machines, machine presses and calendering equipment are used.
A landscape is the visible features of an area of land, its landforms, and how they integrate with natural or human-made features, often considered in terms of their aesthetic appeal. A landscape includes the physical elements of geophysically defined landforms such as mountains, hills, water bodies such as rivers, lakes, ponds and the sea, living elements of land cover including indigenous vegetation, human elements including different forms of land use, buildings, and structures, and transitory elements such as lighting and weather conditions. Combining both their physical origins and the cultural overlay of human presence, often created over millennia, landscapes reflect a living synthesis of people and place that is vital to local and national identity.
Landslides, also known as landslips, rockslips or rockslides, are several forms of mass wasting that may include a wide range of ground movements, such as rockfalls, mudflows, shallow or deep-seated slope failures and debris flows. Landslides occur in a variety of environments, characterized by either steep or gentle slope gradients, from mountain ranges to coastal cliffs or even underwater, in which case they are called submarine landslides.
Language is a structured system of communication that consists of grammar and vocabulary. It is the primary means by which humans convey meaning, both in spoken and signed forms, and may also be conveyed through writing. Human language is characterized by its cultural and historical diversity, with significant variations observed between cultures and across time. Human languages possess the properties of productivity and displacement, which enable the creation of an infinite number of sentences, and the ability to refer to objects, events, and ideas that are not immediately present in the discourse. The use of human language relies on social convention and is acquired through learning.
Lattice may refer to:.
In geography, latitude is a geographic coordinate that specifies the north-south position of a point on the surface of the Earth or another celestial body. Latitude is given as an angle that ranges from −90° at the south pole to 90° at the north pole, with 0° at the Equator. Lines of constant latitude, or parallels, run east-west as circles parallel to the equator. Latitude and longitude are used together as a coordinate pair to specify a location on the surface of the Earth.
Lava is molten or partially molten rock (magma) that has been expelled from the interior of a terrestrial planet or a moon onto its surface. Lava may be erupted at a volcano or through a fracture in the crust, on land or underwater, usually at temperatures from 800 to 1,200 °C. Lava may be erupted directly onto the land surface or onto the sea floor or it may be ejected into the atmosphere before falling back down. The solid volcanic rock resulting from subsequent cooling of the molten material is often also called lava.
Layering can refer to:Layering (horticulture), a means of vegetative propagation
Layering (finance), a strategy in high frequency trading
Layering (linguistics), a principle by which grammaticalisation can be detected
Surface layering, a quasi-crystalline structure at the surfaces of liquids
Layering, a compositional technique in photography
Layering, the use of abstraction layers in software and communication protocol design
Layering, a step in the process of money laundering
Layering, wearing layers of lightweight garments for warmth, known as layered clothing.
Leadership, is defined as the ability of an individual, group, or organization to influence, or guide other individuals, teams, or organizations.
Legislation is the process or result of enrolling, enacting, or promulgating laws by a legislature, parliament, or analogous governing body. Before an item of legislation becomes law it may be known as a bill, and may be broadly referred to as "legislation" while it remains under consideration to distinguish it from other business. Legislation can have many purposes: to regulate, to authorize, to outlaw, to provide (funds), to sanction, to grant, to declare, or to restrict. It may be contrasted with a non-legislative act by an executive or administrative body under the authority of a legislative act.
Lenticular is an adjective often relating to lenses. It may refer to:A term used with two meanings in botany: see Glossary of botanical terms § lenticular
Lenticular cloud, a lens-shaped cloud
Lenticular galaxy, a lens-shaped galaxy
Lenticular (geology), adjective describing a formation with a lens-shaped cross-section
Lenticular nucleus, a lens-shaped nucleus in the brain
Lenticular lens, a technology for making moving or 3D images
Lenticular printing, a technology in which lenticular lenses are used in printing specifically
Lenticular truss bridges, a bridge with a lens-shape truss

.
Lepidoptera or lepidopterans are an order of winged insects which include butterflies and moths. About 180,000 species of the Lepidoptera have been described, representing 10% of the total described species of living organisms, making it the second largest insect order with 126 families and 46 superfamilies, and one of the most widespread and widely recognizable insect orders in the world.
A lexicon is the vocabulary of a language or branch of knowledge. In linguistics, a lexicon is a language's inventory of lexemes. The word lexicon derives from Greek word λεξικόν, neuter of λεξικός meaning 'of or for words'.
In lunar astronomy, libration is the cyclic variation in the apparent position of the Moon that is perceived by observers on the Earth and caused by changes between the orbital and rotational planes of the moon. It causes an observer to see slightly different hemispheres of the surface at different times. It is similar in both cause and effect to the changes in the Moon's apparent size because of changes in distance. It is caused by three mechanisms detailed below, two of which cause a relatively tiny physical libration via tidal forces exerted by the Earth. Such true librations are known as well for other moons with locked rotation.
A lichen is a hybrid colony of algae or cyanobacteria living symbiotically among filaments of multiple fungus species, along with bacteria embedded in the cortex or "skin", in a mutualistic relationship. Lichens are the lifeform that first brought the term symbiosis into biological context.
Lifespan or life span may refer to:Lifespan (film), 1976 film starring Klaus Kinski
Lifespan , 1983 Atari 8-bit computer game
Lifespan (album), 2004 album by Kris Davis
Lifespan: Why We Age – and Why We Don't Have To, 2019 book by David Andrew Sinclair
Lifespan.io, non-profit crowdfunding platform of the Lifespan Extension Advocacy Foundation
Lifespan — former name of Brown University Health, a not-for-profit academic health system in Providence, Rhode Island.
A ligament is a type of fibrous connective tissue in the body that connects bones to other bones. It also connects flight feathers to bones, in dinosaurs and birds. All 30,000 species of amniotes have ligaments.
Lignite, often called brown coal, is a soft, brown, combustible sedimentary rock formed from naturally compressed peat. It has a carbon content around 25–35% and is considered the lowest rank of coal due to its relatively low heat content. When removed from the ground, it contains a very high amount of moisture, which partially explains its low carbon content. Lignite is mined all around the world and is used almost exclusively as a fuel for steam-electric power generation.
Limnology is the study of inland aquatic ecosystems. Pronounced, the name comes from Ancient Greek  λίμνη (límnē) 'lake' and  -λογία (-logía) 'study of'. It includes aspects of the biological, chemical, physical, and geological characteristics of fresh and saline, natural and man-made bodies of water. This includes the study of lakes, reservoirs, ponds, rivers, springs, streams, wetlands, and groundwater. Water systems are often categorized as either running (lotic) or standing (lentic).
Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions.
A lithosphere is the rigid outermost rocky shell of a terrestrial planet or natural satellite. On Earth, it is composed of the crust and the lithospheric mantle, the topmost portion of the upper mantle that behaves elastically on time scales of up to thousands of years or more. The crust and upper mantle are distinguished on the basis of chemistry and mineralogy.
Longitude is a geographic coordinate that specifies the east-west position of a point on the surface of the Earth, or another celestial body. It is an angular measurement, usually expressed in degrees and denoted by the Greek letter lambda (λ). Meridians are imaginary semicircular lines running from pole to pole that connect points with the same longitude. The prime meridian defines 0° longitude; by convention the International Reference Meridian for the Earth passes near the Royal Observatory in Greenwich, south-east London on the island of Great Britain. Positive longitudes are east of the prime meridian, and negative ones are west.
Luminosity is an absolute measure of radiated electromagnetic energy per unit time, and is synonymous with the radiant power emitted by a light-emitting object. In astronomy, luminosity is the total amount of electromagnetic energy emitted per unit of time by a star, galaxy, or other astronomical objects.
A lymphocyte is a type of white blood cell (leukocyte) in the immune system of most vertebrates. Lymphocytes include T cells, B cells, and innate lymphoid cells, of which natural killer cells are an important subtype. They are the main type of cell found in lymph, which prompted the name "lymphocyte". Lymphocytes make up between 18% and 42% of circulating white blood cells.
The Fauna is the whole of animal life present in a particular region or time. The corresponding terms for plants and fungi are flora and funga, respectively. Flora, fauna, funga and other forms of life are collectively referred to as biota. Zoologists and paleontologists use fauna to refer to a typical collection of animals found in a specific time or place, e.g. the "Sonoran Desert fauna" or the "Burgess Shale fauna". Paleontologists sometimes refer to a sequence of faunal stages, which is a series of rocks all containing similar fossils. The study of animals of a particular region is called faunistics.
Magnetism is the class of physical attributes that occur through a magnetic field, which allows objects to attract or repel each other. Because both electric currents and magnetic moments of elementary particles give rise to a magnetic field, magnetism is one of two aspects of electromagnetism.
Magnitude may refer to:.
A mammal is a vertebrate animal of the class Mammalia. Mammals are characterised by the presence of milk-producing mammary glands for feeding their young, a broad neocortex region of the brain, fur or hair, and three middle ear bones. These characteristics distinguish them from reptiles and birds, from which their ancestors diverged in the Carboniferous Period over 300 million years ago. Around 6,640 extant species of mammals have been described and divided into 27 orders. The study of mammals is called mammalogy.
In mathematics, a manifold is a topological space that locally resembles Euclidean space near each point. More precisely, an -dimensional manifold, or -manifold for short, is a topological space with the property that each point has a neighborhood that is homeomorphic to an open subset of -dimensional Euclidean space.
A manuscript was, traditionally, any document written by hand or typewritten, as opposed to mechanically printed or reproduced in some indirect or automated way. More recently, the term has come to be understood to further include any written, typed, or word-processed copy of an author's work, as distinguished from the rendition as a printed version of the same.
Maritime may refer to:.
A massacre is an event of killing defenseless human beings or other animals. It is generally used to describe a targeted mass killing of civilians by an armed group. It can also be used figuratively to refer to a one-sided exchange between armed groups. The word is a loan of a French term for "butchery" or "carnage". Other terms with overlapping scope include war crime, pogrom, mass killing, mass murder, and extrajudicial killing.
In philosophy and metaphysics, materialism is a form of monism holding that matter is the fundamental substance of nature, so that all things, including mind and consciousness, arise from material interactions and depend on physical processes, including those of the human brain and nervous system. It contrasts with monistic idealism, which treats consciousness as fundamental, and is related to naturalism, the view that only natural laws and forces operate in the universe, and to physicalism, the view that all that exists is ultimately physical. Physicalism extends materialism by including forms of physicality beyond ordinary matter, and some use the terms interchangeably.
Mechanism may refer to:Mechanism (economics), a set of rules for a game designed to achieve a certain outcome
Mechanism design, the study of such mechanisms
Mechanism (engineering), rigid bodies connected by joints in order to accomplish a desired force and/or motion transmission
Mechanism (biology), explaining how a feature is created
Mechanism (philosophy), a theory that all natural phenomena can be explained by physical causes
Mechanism (sociology), a theory that all social phenomena can be explained by the existence of a deterministic mechanism.
A megalith is a large stone that has been used to construct a prehistoric structure or monument, either alone or together with other stones. More than 35,000 megalithic structures have been identified across Europe, ranging geographically from Sweden in the north to the Mediterranean Sea in the south.
Melanin is a family of biomolecules organized as oligomers or polymers, which among other functions provide the pigments of many organisms. Melanin pigments are produced in a specialized group of cells known as melanocytes.
A membrane is a selective barrier; it allows some things to pass through but stops others. Such things may be molecules, ions, or other small particles. Membranes can be generally classified into synthetic membranes and biological membranes. Biological membranes include cell membranes ; nuclear membranes, which cover a cell nucleus; and tissue membranes, such as mucosae and serosae. Synthetic membranes are made by humans for use in laboratories and industry.
Metabolism refers to the set of life-sustaining chemical reactions that occur within living organisms. The three main functions of metabolism are the conversion of energy in food into a usable form for cellular processes; the conversion of food to building blocks of macromolecules (biopolymers) such as proteins, lipids, nucleic acids, and some carbohydrates; and the excretion of metabolic wastes. These enzyme-catalyzed reactions allow organisms to grow, reproduce, maintain their structures, and respond to their environments. The word metabolism can also refer to all chemical reactions that occur in living organisms, including digestion and the transportation of substances into and between different cells. In a broader sense, the set of reactions occurring within the cells is called intermediary metabolism.
Metallurgy is a domain of materials science and engineering that studies the physical and chemical behavior of metallic elements, their inter-metallic compounds, and their mixtures, which are known as alloys.
A meteorite is a rock that originated in outer space and has fallen to the surface of a planet or moon. When the original object enters the atmosphere, various factors such as friction, pressure, and chemical interactions with the atmospheric gases cause it to heat up and radiate energy. It then becomes a meteor and forms a fireball, also known as a shooting star; astronomers call the brightest examples "bolides". Once it settles on the larger body's surface, the meteor becomes a meteorite. Meteorites vary greatly in size. For geologists, a bolide is a meteorite large enough to create an impact crater.
Meteorology is the scientific study of the Earth's atmosphere and short-term atmospheric phenomena, with a focus on weather forecasting. It has applications in the military, aviation, energy production, transport, agriculture, construction, weather warnings, and disaster management.
A microorganism, or microbe, is an organism of microscopic size, which may exist in its single-celled form or as a colony of cells. The possible existence of unseen microbial life was suspected from antiquity, with an early attestation in Jain literature authored in 6th-century BC India. The scientific study of microorganisms began with their observation under the microscope in the 1670s by Anton van Leeuwenhoek. In the 1850s, Louis Pasteur found that microorganisms caused food spoilage, debunking the theory of spontaneous generation. In the 1880s, Robert Koch discovered that microorganisms caused the diseases tuberculosis, cholera, diphtheria, and anthrax.
A microclimate refers to localized atmospheric conditions in the near-surface layer, which includes the air immediately above a surface as well as the shallow soil and water environments below it. A microclimate can range in size from a few meters to at most a few kilometers across. It is characterized by a set of persistent, measurable differences in the climate conditions from those in the adjacent surrounding areas. These differences may be subtle or pronounced when evaluated over a diurnal (day-night) or seasonal cycle.
Microfauna are microscopic animals and organisms that exhibit animal-like qualities and have body sizes that are usually <0.1 mm. Microfauna are represented in the animal kingdom and some other heterotrophic, microscopic eukaryotes. A large amount of microfauna are soil microfauna which includes eukaryotic microbes, rotifers, and nematodes. These types of animal-like eukaryotic microbes and true animals are heterotrophic, largely feeding on bacteria. However, some microfauna can consume other things, making them detritivores, fungivores, or even predators.
Weightlessness is the complete or near-complete absence of the sensation of weight, i.e., zero apparent weight. It is also termed zero g-force, or zero-g or, misleadingly, zero gravity.
Microscopy is the technical field of using microscopes to view subjects too small to be seen with the naked eye. There are three well-known branches of microscopy: optical, electron, and scanning probe microscopy, along with the emerging field of X-ray microscopy.
Migration, migratory, or migrate may refer to:.
A militia is a military or paramilitary force that comprises civilian members, as opposed to a professional standing army of regular, full-time military personnel. Militias may be raised in times of need to support regular troops or serve as a pool of available manpower for regular forces to draw from.
Mineralogy is a subject of geology specializing in the scientific study of the chemistry, crystal structure, and physical properties of minerals and mineralized artifacts. Specific studies within mineralogy include the processes of mineral origin and formation, classification of minerals, their geographical distribution, and their utilization.
In visual arts, music, and other media, minimalism is an art movement that emerged in the post-World War II era in Western art. It is often interpreted as a reaction to abstract expressionism and modernism. The movement anticipated various post-minimalist practices in contemporary art that extended or critically reflected on minimalism's original aims. Minimalism emphasized reducing art to its essentials, focusing on the object itself and the viewer's experience with as little mediation from the artist as possible. Prominent artists associated with minimalism include Donald Judd, Agnes Martin, Dan Flavin, Carl Andre, Robert Morris, Anne Truitt, and Frank Stella.
A mirage is a naturally occurring optical phenomenon in which light rays bend via refraction to produce a displaced image of distant objects or the sky. The word comes to English via the French (se) mirer, from the Latin mirari, meaning "to look at, to wonder at".
Mitosis is a part of the cell cycle in eukaryotic cells in which replicated chromosomes are separated into two new nuclei. Cell division by mitosis is an equational division which gives rise to genetically identical cells in which the total number of chromosomes is maintained. Mitosis is preceded by the S phase of interphase and is followed by telophase and cytokinesis, which divide the cytoplasm, organelles, and cell membrane of one cell into two new cells containing roughly equal shares of these cellular components. This process ensures that each daughter cell receives an identical set of chromosomes, maintaining genetic stability across cell generations. The different stages of mitosis altogether define the mitotic phase of a cell cycle—the division of the mother cell into two daughter cells genetically identical to each other.

In chemistry, a mixture is a material made up of two or more different chemical substances which can be separated by physical method. It is an impure substance made up of 2 or more elements or compounds mechanically mixed together in any proportion. A mixture is the physical combination of two or more substances in which the identities are retained and are mixed in the form of solutions, suspensions or colloids.
Signal modulation is the process of varying one or more properties of a periodic waveform in electronics and telecommunication for the purpose of transmitting information.
In Newtonian mechanics, momentum is the product of the mass and velocity of an object. It is a vector quantity, possessing a magnitude and a direction. If m is an object's mass and v is its velocity, then the object's momentum p is:
In the International System of Units (SI), the unit of measurement of momentum is the kilogram metre per second (kg⋅m/s), which is dimensionally equivalent to the newton-second.
A monolith is a geological feature consisting of a single massive stone or rock, such as some mountains. Erosion usually exposes the geological formations, which are often made of very hard and solid igneous or metamorphic rock. Some monoliths are volcanic plugs, solidified lava filling the vent of an extinct volcano.
A monsoon is traditionally a seasonal reversing wind accompanied by corresponding changes in precipitation but is now used to describe seasonal changes in atmospheric circulation and precipitation associated with annual latitudinal oscillation of the Intertropical Convergence Zone (ITCZ) between its limits to the north and south of the equator. Usually, the term monsoon is used to refer to the rainy phase of a seasonally changing pattern, although technically there is also a dry phase. The term is also sometimes used to describe locally heavy but short-term rains.
Morphology, from the Greek and meaning "study of shape", may refer to:.
Mortality may refer to:Fish mortality, a parameter used in fisheries population dynamics to account for the loss of fish in a fish stock through death
Mortality (book), a 2012 collection of essays by Anglo-American writer Christopher Hitchens
Mortality, a property of a Turing machine if it halts when run on any starting configuration
Mortality rate, a measure for the rate at which deaths occur in a given population
Mortality/differential attrition, an error in the internal validity of a scientific study.
Generally, a motif is a recurring element or theme in a work of art or media.
In biology, a mutation is an alteration in the nucleic acid sequence of the genome of an organism, virus, or extrachromosomal DNA. Mutations result from errors during replication, mitosis, meiosis, or damage to DNA, which then may trigger error-prone repair or cause an error during replication. Mutations may also result from substitution, insertion or deletion of segments of DNA due to mobile genetic elements.
Mycology is the branch of biology concerned with the study of fungi, including their taxonomy, genetics, biochemical properties, and use by humans. Fungi can be a source of tinder, food, traditional medicine, as well as entheogens, poison, and infection. Yeasts are among the most heavily utilized members of the fungus kingdom, particularly in food manufacturing.
Myth is a genre of folklore consisting primarily of narratives that play a fundamental role in a society. For scholars, this is totally different from the ordinary sense of the term myth, meaning a belief that is not true, as the veracity of a piece of folklore is entirely irrelevant to determining whether it constitutes a myth.
Nanofibers are fibers with diameters in the nanometer range. Nanofibers can be generated from different polymers and hence have different physical properties and application potentials. Examples of natural polymers include collagen, cellulose, silk fibroin, keratin, gelatin and polysaccharides such as chitosan and alginate. Examples of synthetic polymers include poly(lactic acid) (PLA), polycaprolactone (PCL), polyurethane (PU), poly(lactic-co-glycolic acid) (PLGA), poly(3-hydroxybutyrate-co-3-hydroxyvalerate) (PHBV), and poly(ethylene-co-vinylacetate) (PEVA). Polymer chains are connected via covalent bonds. The diameters of nanofibers depend on the type of polymer used and the method of production. All polymer nanofibers are unique for their large surface area-to-volume ratio, high porosity, appreciable mechanical strength, and flexibility in functionalization compared to their microfiber counterparts.
A nanoparticle or ultrafine particle is a particle of matter 1 to 100 nanometres (nm) in diameter. The term is sometimes used for larger particles, up to 500 nm, or fibers and tubes that are less than 100 nm in only two directions. At the lowest range, metal particles smaller than 1 nm are usually called atom clusters instead.
Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology.
A narrative, story, or tale is any account of a series of related events or experiences, whether non-fictional or fictional. Narratives can be presented through a sequence of written or spoken words, through still or moving images, or through any combination of these.
Nationalism is an ideology or movement that holds that the nation should be congruent with the state. As a movement, it presupposes the existence and tends to promote the interests of a particular nation, especially with the aim of gaining and maintaining its sovereignty (self-determination) over its perceived homeland to create a nation-state. It holds that the nation should govern itself, free from outside interference (self-governance), that a nation is a natural and ideal basis for a polity, and that the nation is the only rightful source of political power. It further aims to build, and maintain, a single national identity, based on a combination of shared social characteristics such as culture, ethnicity, homeland, language, politics, religion, traditions, or belief in a shared singular history, and to promote national unity or solidarity. There are various definitions of a "nation", which leads to different types of nationalism. The two main divergent forms are ethnic nationalism and civic nationalism.
Not to be confused with Naturism.
A nebula is a distinct luminescent part of interstellar medium, which can consist of ionized, neutral, or molecular hydrogen and also cosmic dust. Nebulae are often star-forming regions, such as the Pillars of Creation in the Eagle Nebula. In these regions, the formations of gas, dust, and other materials "clump" together to form denser regions, which attract further matter and eventually become dense enough to form stars. The remaining material is then thought to form planets and other planetary system objects.
Necrosis is a form of cell injury which results in the premature death of cells in living tissue by autolysis. The term "necrosis" came about in the mid-19th century and is commonly attributed to German pathologist Rudolf Virchow, who is often regarded as one of the founders of modern pathology. Necrosis is caused by factors external to the cell or tissue, such as infection, or trauma which result in the unregulated digestion of cell components. In contrast, apoptosis is a naturally occurring programmed and targeted cause of cellular death. While apoptosis often provides beneficial effects to the organism, necrosis is almost always detrimental and can be fatal.
The Neolithic or New Stone Age is an archaeological period, the final division of the Stone Age in Mesopotamia, Asia, Europe and Africa. It saw the Neolithic Revolution, a wide-ranging set of developments that appear to have arisen independently in several parts of the world. This "Neolithic package" included the introduction of farming, domestication of animals, and change from a hunter-gatherer lifestyle to one of settlement. The term 'Neolithic' was coined by John Lubbock in 1865 as a refinement of the three-age system.
In common terminology, a baby is the very young offspring of adult human beings, while infant is a formal or specialised synonym. The terms may also be used to refer to juveniles of other organisms. A newborn is, in colloquial use, a baby who is only hours, days, or weeks old; while in medical contexts, a newborn or neonate is an infant in the first 28 days after birth.
Neoteny, also called juvenilization, is the delaying or slowing of the physiological, or somatic, development of an organism, typically an animal. Neoteny in modern humans is more significant than in other primates. In progenesis or paedogenesis, sexual development is accelerated.
Nephrology is a specialty for both adult internal medicine and pediatric medicine that concerns the study of the kidneys, specifically normal kidney function and kidney disease, the preservation of kidney health, and the treatment of kidney disease, from diet and medication to renal replacement therapy. The word "renal" is an adjective meaning "relating to the kidneys", and its roots are French or late Latin. Whereas according to some opinions, "renal" and "nephro-" should be replaced with "kidney" in scientific writings such as "kidney medicine" or "kidney replacement therapy", other experts have advocated preserving the use of renal and nephro- as appropriate including in "nephrology" and "renal replacement therapy", respectively.
Neuralgia is pain in the distribution of a nerve or nerves, as in intercostal neuralgia, trigeminal neuralgia, and glossopharyngeal neuralgia.
A neutrino is an elementary particle that interacts via the weak interaction and gravity. The neutrino is so named because it is electrically neutral and because its rest mass is so small (-ino) that it was long thought to be zero. The rest mass of the neutrino is much smaller than that of the other known elementary particles .
The weak force has a very short range, the gravitational interaction is extremely weak due to the very small mass of the neutrino, and neutrinos do not participate in the electromagnetic interaction or the strong interaction.
Consequently, neutrinos typically pass through normal matter unimpeded and with no detectable effect.
Niche may refer to:.
Nitrate is a polyatomic ion with the chemical formula NO−3. Salts containing this ion are called nitrates. Nitrates are common components of fertilizers and explosives. Almost all inorganic nitrates are soluble in water. An example of an insoluble (inorganic) nitrate is bismuth oxynitrate.
Nitrogen is a chemical element; it has symbol N and atomic number 7. Nitrogen is a nonmetal and the lightest member of group 15 of the periodic table, often called the pnictogens. It is a common element in the universe, estimated at seventh in total abundance in the Milky Way and the Solar System. At standard temperature and pressure, two atoms of the element bond to form N2, a colourless and odourless diatomic gas. N2 forms about 78% of Earth's atmosphere, making it the most abundant chemical species in air. Because of the volatility of nitrogen compounds, nitrogen is relatively rare in the solid parts of the Earth.
Nomads are communities without fixed habitation who regularly move to and from areas. Such groups include hunter-gatherers, pastoral nomads, tinkers and trader nomads. In the twentieth century, the population of nomadic pastoral tribes slowly decreased, reaching an estimated 30–40 million nomads in the world as of 1995.
In linguistics and semiotics, a notation system is a system of graphics or symbols, characters and abbreviated expressions, used in artistic and scientific disciplines to represent technical facts and quantities by convention. Therefore, a notation is a collection of related symbols that are each given an arbitrary meaning, created to facilitate structured communication within a domain knowledge or field of study.
Nucleus is a Latin word for the seed inside a fruit. It most often refers to:Atomic nucleus, the very dense central region of an atom
Cell nucleus, a central organelle of a eukaryotic cell, containing most of the cell's DNA.
A nutrient is a substance used by an organism to survive, grow and reproduce. The requirement for dietary nutrient intake applies to animals, plants, fungi and protists. Nutrients can be incorporated into cells for metabolic purposes or excreted by cells to create non-cellular structures such as hair, scales, feathers, or exoskeletons. Some nutrients can be metabolically converted into smaller molecules in the process of releasing energy such as for carbohydrates, lipids, proteins and fermentation products leading to end-products of water and carbon dioxide. All organisms require water. Essential nutrients for animals are the energy sources, some of the amino acids that are combined to create proteins, a subset of fatty acids, vitamins and certain minerals. Plants require more diverse minerals absorbed through roots, plus carbon dioxide and oxygen absorbed through leaves. Fungi live on dead or living organic matter and meet nutrient needs from their host.
Obsidian is a naturally occurring volcanic glass formed when lava extruded from a volcano cools rapidly with minimal crystal growth. It is an igneous rock. Produced from felsic lava, obsidian is rich in the lighter elements such as silicon, oxygen, aluminium, sodium, and potassium. It is commonly found within the margins of rhyolitic lava flows known as obsidian flows. These flows have a high content of silica, giving them a high viscosity. The high viscosity inhibits the diffusion of atoms through the lava, which inhibits the first step (nucleation) in the formation of mineral crystals. Together with rapid cooling, this results in a natural glass forming from the lava.
Oceanography, also known as oceanology, sea science, ocean science, and marine science, is the scientific study of the ocean, including its physics, chemistry, biology, and geology.
The sense of smell, or olfaction, is the special sense through which smells are perceived. The sense of smell has many functions, including detecting desirable foods, hazards, and pheromones, and plays a role in taste.
Oligarchy is a form of government in which power rests with a small number of people. Leaders of such regimes are often referred to as oligarchs, and generally are characterized by having titles of nobility or high amounts of wealth.
An oligopoly is a market in which pricing control lies in the hands of a few sellers.
An ombudsman is a government official who investigates and tries to resolve complaints, usually through recommendations or mediation. They are usually appointed by the government or by parliament.

Ontology is the philosophical study of being. It is traditionally understood as the subdiscipline of metaphysics focused on the most general features of reality. As one of the most fundamental concepts, being encompasses all of reality and every entity within it. To articulate the basic structure of being, ontology examines the commonalities among all things and investigates their classification into basic types, such as the categories of particulars and universals. Particulars are unique, non-repeatable entities, such as the person Socrates, whereas universals are general, repeatable entities, like the color green. Another distinction exists between concrete objects existing in space and time, such as a tree, and abstract objects existing outside space and time, like the number 7. Systems of categories aim to provide a comprehensive inventory of reality by employing categories such as substance, property, relation, state of affairs, and event.
Opacity is the measure of impenetrability to electromagnetic or other kinds of radiation, especially visible light. In radiative transfer, it describes the absorption and scattering of radiation in a medium, such as a plasma, dielectric, shielding material, glass, etc. An opaque object is translucent. When light strikes an interface between two substances, in general, some may be reflected, some absorbed, some scattered, and the rest transmitted.
In genetics, an operon is a functioning unit of DNA containing a cluster of genes under the control of a single promoter. The genes are transcribed together into an mRNA strand and either translated together in the cytoplasm, or undergo splicing to create monocistronic mRNAs that are translated separately, i.e. several strands of mRNA that each encode a single gene product. The result of this is that the genes contained in the operon are either expressed together or not at all. Several genes must be co-transcribed to define an operon.
Optics is the branch of physics that studies the behaviour, manipulation, and detection of electromagnetic radiation, including its interactions with matter and instruments that use or detect it. Optics usually describes the behaviour of visible, ultraviolet, and infrared light. The study of optics extends to other forms of electromagnetic radiation, including radio waves, microwaves,
and X-rays. The term optics is also applied to technology for manipulating beams of elementary charged particles.
In celestial mechanics, an orbit is the curved trajectory of an object under the influence of an attracting force. Alternatively, it is known as an orbital revolution, because it is a rotation around an axis external to the moving body. Examples for orbits include the trajectory of a planet around a star, a natural satellite around a planet, or an artificial satellite around an object or position in space such as a planet, moon, asteroid, or Lagrange point. Normally, orbit refers to a regularly repeating trajectory, although it may also refer to a non-repeating trajectory. To a close approximation, planets and satellites follow elliptic orbits, with the center of mass being orbited at a focal point of the ellipse, as described by Kepler's laws of planetary motion.
An organism is any living thing that functions as an individual. Such a definition raises more problems than it solves, not least because the concept of an individual is also difficult. Several criteria, few of which are widely accepted, have been proposed to define what constitutes an organism. Among the most common is that an organism has autonomous reproduction, growth, and metabolism. This would exclude viruses, even though they evolve like organisms.
Osmosis is the spontaneous net movement of solvent molecules through a selectively permeable membrane from a region of high water potential to a region of low water potential, in the direction that tends to equalize the solute concentrations on the two sides. It may also be used to describe a physical process in which any solvent moves across a selectively permeable membrane separating two solutions of different concentrations. Osmosis can be made to do work. Osmotic pressure is defined as the external pressure required to prevent net movement of solvent across the membrane. Osmotic pressure is a colligative property, meaning that the osmotic pressure depends on the molar concentration of the solute but not on its identity. Osmotic transport occurs through viscous flow of the solvent under a pressure gradient.
In epidemiology, an outbreak is a sudden increase in occurrences of a disease when cases are in excess of normal expectancy for the location or season. It may affect a small and localized group or impact upon thousands of people across an entire continent. The number of cases varies according to the disease-causing agent, and the size and type of previous and existing exposure to the agent. Outbreaks include many epidemics, which is a term normally only used for infectious diseases, as well as diseases with an environmental origin, such as a water or foodborne disease. They may affect a region in a country or a group of countries. Pandemics are near-global disease outbreaks when multiple and various countries around the Earth are soon infected.
Redox is a type of chemical reaction in which the oxidation states of the reactants change. Oxidation is the loss of electrons or an increase in the oxidation state, while reduction is the gain of electrons or a decrease in the oxidation state. The oxidation and reduction processes occur simultaneously in the chemical reaction.
Paleoclimatology is the scientific study of climates predating the invention of meteorological instruments, when no direct, artificial measurement data were available. As instrumental records only span a tiny part of Earth's history, the reconstruction of ancient climate is important to understand natural variation and the evolution of the current climate.
Palaeography (UK) or paleography (US) is the study and academic discipline of historical writing systems. It encompasses the historicity of manuscripts and texts, subsuming deciphering and dating of historical manuscripts, as well as the analysis of historic penmanship, handwriting scripts, signification, and printed media.
Paleontology or palaeontology is the scientific study of the life of the past, mainly but not exclusively through the study of fossils. Paleontologists use fossils as a means to classify organisms, measure geologic time, and assess the interactions between prehistoric organisms and their natural environment. While paleontological observations are known from at least the 6th century BC, the foundation of paleontology as a science dates back to the work of Georges Cuvier in 1796. Cuvier demonstrated evidence for the concept of extinction and how the life of the past was not necessarily the same as that of the present. The field developed rapidly over the course of the following decades, and the French word paléontologie was introduced for the study in 1822, which was derived from the Ancient Greek word for 'ancient' and words describing relatedness and a field of study. Further advances in the field accompanied the work of Charles Darwin who popularized the concept of evolution. Together, evolution and extinction can be understood as complementary processes that shaped the history of life.
A pandemic is an epidemic of an infectious disease that has a sudden increase in cases and spreads across a large region, for instance multiple continents or worldwide, affecting a substantial portion of the human population. Widespread endemic diseases with a stable number of infected individuals such as recurrences of seasonal influenza are generally excluded as they occur simultaneously in large regions of the globe rather than being spread worldwide.
A panorama is any wide-angle view or representation of a physical space, whether in painting, drawing, photography, film, seismic images, or 3D modeling. The word was coined in the 18th century by the English painter Robert Barker to describe his panoramic paintings of Edinburgh and London. The motion-picture term panning is derived from panorama.
A paradox is a logically self-contradictory statement or a statement that runs contrary to one's expectation. It is a statement that, despite apparently valid reasoning from true or apparently true premises, leads to a seemingly self-contradictory or a logically unacceptable conclusion. A paradox usually involves contradictory-yet-interrelated elements that exist simultaneously and persist over time. They result in "persistent contradiction between interdependent elements" leading to a lasting "unity of opposites".
Parallax is a displacement or difference in the apparent position of an object viewed along two different lines of sight and is measured by the angle or half-angle of inclination between those two lines. Due to foreshortening, nearby objects show a larger parallax than farther objects, so parallax can be used to determine distances.
Parasitism is a close relationship between species, where one organism, the parasite, lives on or inside another organism, the host, causing it some harm, and is adapted structurally to this way of life. The entomologist E. O. Wilson characterised parasites' way of feeding as "predators that eat prey in units of less than one". Parasites include single-celled protozoans such as the agents of malaria, sleeping sickness, and amoebic dysentery; animals such as hookworms, lice, mosquitoes, and vampire bats; fungi such as honey fungus and the agents of ringworm; and plants such as mistletoe, dodder, and the broomrapes.
In the physical sciences, a particle is a small localized object which can be described by several physical or chemical properties, such as volume, density, or mass. They vary greatly in size or quantity, from subatomic particles like the electron, to microscopic particles like atoms and molecules, to macroscopic particles like powders and other granular materials. Particles can also be used to create scientific models of even larger objects depending on their density, such as humans moving in a crowd or celestial bodies in motion.
In biology, a pathogen, in the oldest and broadest sense, is any organism, agent or micro-organism that can produce disease. A pathogen may also be referred to as an infectious agent, or simply a germ.
Pathology is the study of disease. The word pathology also refers to the study of disease in general, incorporating a wide range of biology research fields and medical practices. However, when used in the context of modern medical treatment, the term is often used in a narrower fashion to refer to processes and tests that fall within the contemporary medical field of "general pathology", an area that includes a number of distinct but inter-related medical specialties that diagnose disease, mostly through analysis of tissue and human cell samples. Pathology is a significant field in modern medical diagnosis and medical research. A physician practicing pathology is called a pathologist.
Pedagogy, most commonly understood as the approach to teaching, is the theory and practice of learning, and how this process influences, and is influenced by, the social, political, and psychological development of learners. Pedagogy, taken as an academic discipline, is the study of how knowledge and skills are imparted in an educational context, and it considers the interactions that take place during learning. Both the theory and practice of pedagogy vary greatly as they reflect different social, political, and cultural contexts.
A peninsula is a landform that extends from a mainland, is connected to the mainland on only one side, and is mostly surrounded by water. Peninsulas exist on each continent. The largest peninsula in the world is the Arabian Peninsula.
In geometry, a pentagon is any five-sided polygon or 5-gon. The sum of the internal angles in a simple pentagon is 540°.
Peptides are short chains of amino acids linked by peptide bonds. A polypeptide is a longer, continuous, unbranched peptide chain. Polypeptides that have a molecular mass of 10,000 Da or more are called proteins. Chains of fewer than twenty amino acids are called oligopeptides, and include dipeptides, tripeptides, and tetrapeptides. Proteins are polypeptides, i.e. large peptides.
Perception is the organization, identification, and interpretation of sensory information, in order to represent and understand the presented information or environment. All perception involves signals that go through the nervous system, which in turn result from physical or chemical stimulation of the sensory system. Vision involves light striking the retina of the eye; smell is mediated by odor molecules; and hearing involves pressure waves.
In physics, chemistry, and materials science, percolation refers to the movement and filtering of fluids through porous materials. It is not described by Darcy's law. Broader applications have since been developed that cover connectivity of many systems modeled as lattices or graphs, analogous to connectivity of lattice components in the filtration problem that modulates capacity for percolation.
Permafrost is soil or underwater sediment which continuously remains below 0 °C (32 °F) for two years or more; the oldest permafrost has been continuously frozen for around 700,000 years. Whilst the shallowest permafrost has a vertical extent of below a meter (3 ft), the deepest is greater than 1,500 m (4,900 ft). Similarly, the area of individual permafrost zones may be limited to narrow mountain summits or extend across vast Arctic regions. The ground beneath glaciers and ice sheets is not usually defined as permafrost, so on land, permafrost is generally located beneath a so-called active layer of soil which freezes and thaws depending on the season.
Permeability, permeable, and semipermeable may refer to:.
In mathematics, a permutation of a set can mean one of two different things:an arrangement of its members in a sequence or linear order, or
the act or process of changing the linear order of an ordered set.
Pesticides are substances that are used to control pests. They include herbicides, insecticides, nematicides, fungicides, and many others. The most common of these are herbicides, which account for approximately 50% of all pesticide use globally. Most pesticides are used as plant protection products, which in general protect plants from weeds, fungi, or insects.
Petrology is the branch of geology that studies rocks, their mineralogy, composition, texture, structure and the conditions under which they form. Petrology has three subdivisions: igneous, metamorphic, and sedimentary petrology. Igneous and metamorphic petrology are commonly taught together because both make heavy use of chemistry, chemical methods, and phase diagrams. Sedimentary petrology is commonly taught together with stratigraphy because it deals with the processes that form sedimentary rock. Modern sedimentary petrology is making increasing use of chemistry.
Phagocytes are cells that protect the body by ingesting harmful foreign particles, bacteria, and dead or dying cells. They include monocytes, macrophages, neutrophils, tissue dendritic cells, and mast cells. Their name comes from the Greek phagein, "to eat" or "devour", and "-cyte", the suffix in biology denoting "cell", from the Greek kutos, "hollow vessel". They are essential for fighting infections and for subsequent immunity. Phagocytes are important throughout the animal kingdom and are highly developed within vertebrates. One litre of human blood contains about six billion phagocytes. They were discovered in 1882 by Ilya Ilyich Mechnikov while he was studying starfish larvae. Mechnikov was awarded the 1908 Nobel Prize in Physiology or Medicine for his discovery. Phagocytes occur in many species; some amoebae behave like macrophage phagocytes, which suggests that phagocytes appeared early in the evolution of life.
Pharmacology is the science of drugs and medications, including a substance's origin, composition, pharmacokinetics, pharmacodynamics, therapeutic use, and toxicology. More specifically, it is the study of the interactions that occur between a living organism and chemicals that affect normal or abnormal biochemical function. If substances have medicinal properties, they are considered pharmaceuticals.
In genetics, the phenotype is the set of observable characteristics or traits of an organism. The term covers all traits of an organism other than its genome, however transitory: the organism's morphology, its developmental processes, its biochemical and physiological properties whether reversible or irreversible, and all its behavior, such as a peacock's display.
A phoneme is a set of similar speech sounds that are perceptually regarded by the speakers of a language as a single basic sound—a smallest possible phonetic unit—that helps distinguish one word from another. All languages contain phonemes, and all spoken languages include both consonant and vowel phonemes. Phonemes are studied under phonology, a branch of linguistics.
A photon is an elementary particle that is a quantum of the electromagnetic field, including electromagnetic radiation such as light and radio waves, and the force carrier for the electromagnetic force. Photons are massless particles that can only move at one speed, the speed of light measured in a vacuum. The photon belongs to the class of boson particles.
The photosphere is a star's outer shell from which light is radiated. It extends into a star's surface until the plasma becomes opaque, equivalent to an optical depth of approximately 2⁄3, or equivalently, a depth from which 50% of light will escape without being scattered.
In biology, a phylum is a level of classification, or taxonomic rank, that is below kingdom and above class. Traditionally, in botany the term division has been used instead of phylum, although the International Code of Nomenclature for algae, fungi, and plants accepts the terms as equivalent. Depending on definitions, the animal kingdom Animalia contains about 32 phyla, the plant kingdom Plantae contains about 14 phyla, and the fungus kingdom Fungi contains about eight phyla. Current research in phylogenetics is uncovering the relationships among phyla within larger clades like Ecdysozoa and Embryophyta.
A pinnacle is an architectural element originally forming the cap or crown of a buttress or small turret, but afterwards used on parapets at the corners of towers and in many other situations. The pinnacle looks like a small spire. It was mainly used in Gothic architecture.
A pipeline is a system of pipes for long-distance transportation of a liquid or gas, typically to a market area for consumption. Industry datasets indicate the length of the global trunk/transmission pipeline network is on the order of ~2.19 million km by 2025, with North America accounting for ~44%. The United States had 65%, Russia had 8%, and Canada had 3%, thus 76% of all pipeline were in these three countries. The main attribute to pollution from pipelines is caused by corrosion and leakage.
A piston is a component of reciprocating engines, reciprocating pumps, gas compressors, hydraulic cylinders and pneumatic cylinders, among other similar mechanisms. It is the moving component that is contained by a cylinder and is made gas-tight by piston rings. In an engine, its purpose is to transfer force from expanding gas in the cylinder to the crankshaft via a piston rod and/or connecting rod. In a pump, the function is reversed and force is transferred from the crankshaft to the piston for the purpose of compressing or ejecting the fluid in the cylinder. In some engines, the piston also acts as a valve by covering and uncovering ports in the cylinder.
Plankton are organisms that drift in water but are unable to actively propel themselves against currents. Marine plankton include drifting organisms that inhabit the saltwater of oceans and the brackish waters of estuaries. Freshwater plankton are similar to marine plankton, but are found in lakes and rivers. An individual plankton organism in the plankton is called a plankter. In the ocean plankton provide a crucial source of food, particularly for larger filter-feeding animals, such as bivalves, sponges, forage fish and baleen whales.
In geology and physical geography, a plateau, also called a high plain or a tableland, is an area of highland consisting of flat terrain that is raised sharply above the surrounding area on at least one side. Often one or more sides have deep escarpments or hills. Plateaus can be formed by a number of processes, including upwelling of volcanic magma, extrusion of lava, and erosion by water and glaciers. Plateaus are classified according to their surrounding environment as intermontane, piedmont, or continental. A few plateaus may have a small flat top while others have wider ones.
The Pleistocene is the geological epoch that lasted from c. 2.58 million to 11,700 years ago, spanning the Earth's most recent period of repeated glaciations. Before a change was finally confirmed in 2009 by the International Union of Geological Sciences, the cutoff of the Pleistocene and the preceding Pliocene was regarded as being 1.806 million years Before Present (BP). Publications from earlier years may use either definition of the period. The end of the Pleistocene corresponds with the end of the last glacial period and also with the end of the Paleolithic age used in archaeology. The name comes from Ancient Greek πλεῖστος (pleîstos), meaning "most", and καινός (kainós), meaning "new, recent".
The Pliocene is the epoch in the geologic time scale that extends from 5.33 to 2.58 million years ago (Ma). It is the second and most recent epoch of the Neogene Period in the Cenozoic Era. The Pliocene follows the Miocene Epoch and is followed by the Pleistocene Epoch. Prior to the 2009 revision of the geologic time scale, which placed the four most recent major glaciations entirely within the Pleistocene, the Pliocene also included the Gelasian Stage, which lasted from 2.59 to 1.81 Ma, and is now included in the Pleistocene.
Plutonium is a chemical element; it has symbol Pu and atomic number 94. It is a silvery-gray actinide metal that tarnishes when exposed to air, and forms a dull coating when oxidized. The element normally exhibits six allotropes and four oxidation states. It reacts with carbon, halogens, nitrogen, silicon and hydrogen. When exposed to moist air, it forms oxides and hydrides that can expand the sample up to 70% in volume, which in turn flake off as a powder that is pyrophoric. It is radioactive and can accumulate in bones, which makes the handling of plutonium dangerous.
Pneumatics is the use of gas or pressurized air to create mechanical motion in mechanical systems.
Polarization or polarisation may refer to:.
A polymer is a substance or material that consists of very large molecules, or macromolecules, that are constituted by many repeating subunits derived from one or more species of monomers. Due to their broad spectrum of properties, both synthetic and natural polymers play essential and ubiquitous roles in everyday life. Polymers range from familiar synthetic plastics such as polystyrene to natural biopolymers such as DNA and proteins that are fundamental to biological structure and function. Polymers, both natural and synthetic, are created via polymerization of many small molecules, known as monomers. Their consequently large molecular mass, relative to small molecule compounds, produces unique physical properties including toughness, high elasticity, viscoelasticity, and a tendency to form amorphous and semicrystalline structures rather than crystals.
In biology, a population of organisms is a group of individuals of the same species, defined by a discontinuity or disjunction from other groups of individuals in certain characteristics, such as living area, genetic attributes, demographic structure. Among biologists, the term definition varies, in some cases significantly, and sometimes those variations can be confusing. There are also plenty of other terms to describe groups of individuals if no clear disjunction is present. Commonly, a population can be described by what individuals constitute the population, its size, a geographical area it occupies, and the time within which the population is examined. In qualitative terms, it is usually defined like "a group of organisms of the same species occupying a particular space at a particular time".
Porosity or void fraction is a measure of the void spaces in a material, and is a fraction of the volume of voids over the total volume, between 0 and 1, or as a percentage between 0% and 100%. Strictly speaking, some tests measure the "accessible void", the total amount of void space accessible from the surface.
Portfolio may refer to:.
Positioning may refer to:Positioning (marketing), creating an identity in the minds of a target market
Positioning theory, a theory in social psychology
Positioning, reader context
Positioning (telecommunications), a technology to approximate where a mobile phone temporarily resides
Grappling position, the positioning and holds of combatants engaged in grappling
Geopositioning, determining the location of an object in space.
Pottery is the process and the products of forming vessels and other objects with clay and other raw materials, which are fired at high temperatures to give them a hard and durable form. The place where such wares are made by a potter is also called a pottery. The definition of pottery, used by the ASTM International, is "all fired ceramic wares that contain clay when formed, except technical, structural, and refractory products". End applications include tableware, decorative ware, sanitary ware, and in technology and industry such as electrical insulators and laboratory ware. In art history and archaeology, especially of ancient and prehistoric periods, pottery often means only vessels, and sculpted figurines of the same material are called terracottas.
In an aqueous solution, precipitation is the "sedimentation of a solid material from a liquid solution". The solid formed is called the precipitate. In case of an inorganic chemical reaction leading to precipitation, the chemical reagent causing the solid to form is called the precipitant.
Prehistory, sometimes referred to as pre-literary history, is the period of human history between the first known use of stone tools by hominins c. 3.3 million years ago and the beginning of recorded history with the invention of writing systems. The use of symbols, marks, and images appears very early among humans, but the earliest known writing systems appeared c. 5,200 years ago. The adoption of writing across the globe has been a slow process, so that the end of prehistory occurred at different times in different places, and the term is less often used in discussing societies where prehistory ended relatively recently. The period when a culture is written about by others, but has not developed its own writing system, is often known as the protohistory of the culture.
Pressure is the force applied perpendicular to the surface of an object per unit area over which that force is distributed. Gauge pressure is the pressure relative to the ambient pressure.
Primatology is the scientific study of primates. Unlike branches of zoology focused on specific animal groups, primatology – and the primate order — includes both human and nonhuman animals. Thus, the field entails significant overlap with anthropology, the study of humans, and related sciences.
Probability concerns events and numerical descriptions of how likely they are to occur. The probability of an event is a number between 0 and 1; the larger the probability, the more likely an event is to occur. This number is often expressed as a percentage (%), ranging from 0% to 100%. A simple example is the tossing of a fair (unbiased) coin. Since the coin is fair, the two outcomes are both equally probable; the probability of "heads" equals the probability of "tails"; and since no other outcomes are possible, the probability of either "heads" or "tails" is 1/2.
Procedure may refer to:Medical procedure
Instructions or recipes, a set of commands that show how to achieve some result, such as to prepare or make something
Procedure (business), specifying parts of a business process
Standard operating procedure, a step-by-step instruction to achieve some result, used in industry and military
Legal procedure, the body of law and rules used in the administration of justice in the court system, including:
Civil procedure
Criminal procedure
Administrative procedure
Parliamentary procedure, a set of rules governing meetings
Procedure, also termed a subroutine, function, or subprogram
Stored procedure, a subroutine in the data dictionary of a relational database
The Procedure, an American hardcore band

.
Propagation can refer to:Chain propagation in a chemical reaction mechanism
Crack propagation, the growth of a crack during the fracture of materials
Propaganda, non-objective information used to further an agenda
Reproduction, and other forms of multiplication or increase
Plant propagation, the production of more plants
Propagation of schema, in artificial reproduction
Software propagation, the distribution of free software
Wave propagation, the motion of a wave
Radio propagation, the application of wave propagation to radio communicationsIn musicPropagation (album)
"Propagation", a song by Lower Dens from the album Nootropics
"Propagation", a song by Com Truise from the album Iteration

.
Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.
Protocol may refer to:.
A proton is a stable subatomic particle, symbol p, H+, or 1H+ with a positive electric charge of +1 e (elementary charge). Its mass is slightly less than the mass of a neutron and approximately 1836 times the mass of an electron (the proton-to-electron mass ratio). Protons and neutrons, each with a mass of approximately one dalton, are jointly referred to as nucleons (particles present in atomic nuclei).
Provenance is the chronology of the ownership, custody or location of a historical object. The term was originally mostly used in relation to works of art, but is now used in similar senses in a wide range of fields, including archaeology, paleontology, archival science, economy, computing, and scientific enquiry in general.
In computer science, pseudocode is a description of the steps in an algorithm using a mix of conventions of programming languages with informal, usually self-explanatory, notation of actions and conditions. Although pseudocode shares features with regular programming languages, it is intended for human reading rather than machine control. Pseudocode typically omits details that are essential for machine implementation of the algorithm, meaning that pseudocode can only be verified by hand. The programming language is augmented with natural language description details, where convenient, or with compact mathematical notation. The reasons for using pseudocode are that it is easier for people to understand than conventional programming language code and that it is an efficient and environment-independent description of the key principles of an algorithm. It is commonly used in textbooks and scientific publications to document algorithms and in planning of software and other algorithms.
Pseudoscience consists of statements, beliefs, or practices that claim to be scientific or factual but are inherently incompatible with the scientific method. Pseudoscience is often characterized by contradictory, exaggerated or unfalsifiable claims; reliance on confirmation bias rather than rigorous attempts at refutation; lack of openness to evaluation by other experts; absence of systematic practices when developing hypotheses; and continued adherence long after the pseudoscientific hypotheses have been experimentally discredited. It is not the same as junk science.
Psychometrics is a field of study within psychology concerned with the theory and technique of measurement. Psychometrics generally covers specialized fields within psychology and education devoted to testing, measurement, assessment, and related activities. Psychometrics is concerned with the objective measurement of latent constructs that cannot be directly observed. Examples of latent constructs include intelligence, personality factors, mental disorders, and educational achievement. The levels of individuals on nonobservable latent variables are inferred through mathematical modeling based on what is observed from individuals' responses to items on tests and scales.
Pterosaurs are an extinct clade of flying reptiles in the order Pterosauria. They existed during most of the Mesozoic: from the Late Triassic to the end of the Cretaceous. Pterosaurs are the earliest vertebrates known to have evolved powered flight. Their wings were formed by a membrane of skin, muscle, and other tissues stretching from the ankles to a dramatically lengthened fourth finger.
A pulley is a wheel on an axle or shaft enabling a taut cable or belt passing over the wheel to move and change direction, or transfer power between itself and a shaft.
In physics, a quantum is the minimum amount of any physical entity involved in an interaction. The fundamental notion that a property can be "quantized" is referred to as "the hypothesis of quantization". This means that the magnitude of the physical property can take on only discrete values consisting of integer multiples of one quantum. For example, a photon is a single quantum of light of a specific frequency. Similarly, the energy of an electron bound within an atom is quantized and can exist only in certain discrete values. Atoms and matter in general are stable because electrons can exist only at discrete energy levels within an atom. Quantization is one of the foundations of the much broader physics of quantum mechanics. Quantization of energy and its influence on how energy and matter interact is part of the fundamental framework for understanding and describing nature.
A quarantine is a restriction on the movement of people, animals, and goods which is intended to prevent the spread of disease or pests. It is often used in connection to disease and illness, preventing the movement of those who may have been exposed to a communicable disease, yet do not have a confirmed medical diagnosis. It is distinct from medical isolation, in which those confirmed to be infected with a communicable disease are isolated from the healthy population.
A quarry is a type of rock and earth materials—like limestone, granite, marble, sand, and gravel—directly from the surface to use in building and construction. The operation of quarries is regulated in some jurisdictions to manage their safety risks and reduce their environmental impact.
A quasar is an extremely luminous active galactic nucleus (AGN). It is sometimes known as a quasi-stellar object, abbreviated QSO. The emission from an AGN is powered by accretion onto a supermassive black hole with a mass ranging from millions to tens of billions of solar masses, surrounded by a gaseous accretion disc. Gas in the disc falling towards the black hole heats up and releases energy in the form of electromagnetic radiation. The radiant energy of quasars is enormous; the most powerful quasars have luminosities thousands of times greater than that of a galaxy such as the Milky Way. Quasars are usually categorized as a subclass of the more general category of AGN. The redshifts of quasars are of cosmological origin.
A quorum is the minimum number of members of a group necessary to constitute the group at a meeting. In a deliberative assembly, a quorum is necessary to conduct the business of that group. In contrast, a plenum is a meeting of the full body. A body, or a meeting or vote of it, is quorate if a quorum is present.
In physics, radiation is the emission or transmission of energy in the form of waves or particles through space or a material medium. This includes:electromagnetic radiation consisting of photons, such as radio waves, microwaves, infrared, visible light, ultraviolet, x-rays, and gamma radiation (γ)
particle radiation consisting of particles of non-zero rest energy, such as alpha radiation (α), beta radiation (β), proton radiation and neutron radiation
acoustic radiation, such as ultrasound, sound, and seismic waves, all dependent on a physical transmission medium
gravitational radiation, in the form of gravitational waves, ripples in spacetime.
Radiometry is a set of techniques for measuring electromagnetic radiation, including visible light. Radiometric techniques in optics characterize the distribution of the radiation's power in space, as opposed to photometric techniques, which characterize the light's interaction with the human eye.
Rainforests are forests characterized by a closed and continuous tree canopy, moisture-dependent vegetation, the presence of epiphytes and lianas and the absence of wildfire. Rainforests can be generally classified as tropical rainforests or temperate rainforests, but other types have been described.
Reaction may refer to a process or to a response to an action, event, or exposure.
Reactor may refer to:.
In chemistry, a reagent or analytical reagent is a substance or compound added to a system to cause a chemical reaction, or test if one occurs. A reactant is a substance or compound that is consumed in a chemical reaction. The terms reactant and reagent are often used interchangeably, but reactant specifies a substance consumed in the course of a chemical reaction; reagent is used in the context of chemical analysis, while reactant is used in the context of reaction itself. Solvents, though involved in the reaction mechanism, are usually not called reactants. Similarly, catalysts are not consumed by the reaction, so they are not reactants. In biochemistry, especially in connection with enzyme-catalyzed reactions, the reactants are commonly called substrates.
In economics, a recession is a business cycle contraction that occurs when there is a period of broad decline in economic activity. Recessions generally occur when there is a widespread drop in spending. This may be triggered by various events, such as a financial crisis, an external trade shock, an adverse supply shock, the bursting of an economic bubble, or a large-scale anthropogenic or natural disaster. There is no official definition of a recession, according to the International Monetary Fund.
Reconstruction may refer to:.
A reef is a ridge or shoal of rock, coral, or similar relatively stable material lying beneath the surface of a natural body of water. Many reefs result from natural, abiotic (non-living) processes such as deposition of sand or wave erosion planing down rock outcrops. However, reefs such as the coral reefs of tropical waters are formed by biotic (living) processes, dominated by corals and coralline algae. Artificial reefs, such as shipwrecks and other man-made underwater structures, may occur intentionally or as the result of an accident. These are sometimes designed to increase the physical complexity of featureless sand bottoms to attract a more diverse range of organisms. They provide shelter to various aquatic animals which help prevent extinction. Another reason reefs are put in place is for aquaculture, and fish farmers who are looking to improve their businesses sometimes invest in them. Reefs are often quite near to the surface, but not all definitions require this.
In physics, refraction is the redirection of a wave as it passes from one medium to another. The redirection can be caused by the wave's change in speed or by a change in the medium. Refraction of light is the most commonly observed phenomenon, but other waves such as sound waves and water waves also experience refraction. How much a wave is refracted is determined by the change in wave speed and the initial direction of wave propagation relative to the direction of change in speed.
Regolith is a blanket of unconsolidated, loose, heterogeneous superficial deposits covering solid rock. It includes dust, broken rocks, and other related materials and is present on Earth, the Moon, Mars, some asteroids, and other terrestrial planets and moons.
Regression or regressions may refer to:.
In behavioral psychology, reinforcement refers to consequences that increase the likelihood of an organism's future behavior, typically in the presence of a particular antecedent stimulus. For example, a rat can be trained to push a lever to receive food whenever a light is turned on; in this example, the light is the antecedent stimulus, the lever pushing is the operant behavior, and the food is the reinforcer. Likewise, a student that receives attention and praise when answering a teacher's question will be more likely to answer future questions in class; the teacher's question is the antecedent, the student's response is the behavior, and the praise and attention are the reinforcements. Punishment is the inverse to reinforcement, referring to any behavior that decreases the likelihood that a response will occur. In operant conditioning terms, punishment does not need to involve any type of pain, fear, or physical actions; even a brief spoken expression of disapproval is a type of punishment.
Relativity may refer to:.
Remedy, Remedies, The Remedy or Remediation may refer to:.
No summary available.
Reproduction is the biological process by which new individual organisms – offspring – are produced from their parent or parents. There are two forms of reproduction: asexual and sexual.
A reservoir is an enlarged lake behind a dam, usually built to store fresh water, often doubling for hydroelectric power generation.
Resonance is a phenomenon that occurs when an object or system is subjected to an external force or vibration whose frequency matches a resonant frequency of the system, defined as a frequency that generates a maximum amplitude response in the system. When this happens, the object or system absorbs energy from the external force and starts vibrating with a larger amplitude. Resonance can occur in various systems, such as mechanical, electrical, or acoustic systems, and it is often desirable in certain applications, such as musical instruments or radio receivers. However, resonance can also be detrimental, leading to excessive vibrations or even structural failure in some cases.
Respiration may refer to:.
Restoration is the act of restoring something to its original state. This may refer to:Conservation and restoration of cultural property
Audio restoration
Conservation and restoration of immovable cultural property
Film restoration
Image restoration
Textile restoration
Ecological restoration.
The retina is the innermost, light-sensitive layer of tissue of the eye of most vertebrates and some molluscs. The optics of the eye create a focused two-dimensional image of the visual world on the retina, which then processes that image within the retina and sends nerve impulses along the optic nerve to the visual cortex to create visual perception. The retina serves a function which is in many ways analogous to that of the film or image sensor in a camera.
In political science, a revolution is a rapid, fundamental transformation of a society's class, state, ethnic or religious structures. According to sociologist Jack Goldstone, all revolutions contain "a common set of elements at their core: (a) efforts to change the political regime that draw on a competing vision of a just order, (b) a notable degree of informal or formal mass mobilization, and (c) efforts to force change through noninstitutionalized actions such as mass demonstrations, protests, strikes, or violence.".
Rheology is the study of the flow of matter, primarily in a fluid state, as well as "soft solids", which experience conditions under which they respond with plastic flow rather than elastic deformation to forces applied to them. Rheology is the branch of physics that deals with the deformation and flow of materials, both solids and liquids.
Otorhinolaryngology is a surgical subspecialty within medicine that deals with the surgical and medical management of conditions of the head and neck. Doctors who specialize in this area are called otorhinolaryngologists, otolaryngologists, head and neck surgeons, or ENT surgeons or physicians.
In geology, a rift is a linear zone where the lithosphere is being pulled apart and is an example of extensional tectonics. Typical rift features are a central linear downfaulted depression, called a graben, or more commonly a half-graben with normal faulting and rift-flank uplifts mainly on one side. Where rifts remain above sea level they form a rift valley, which may be filled by water forming a rift lake. The axis of the rift area may contain volcanic rocks, and active volcanism is a part of many, but not all, active rift systems.
Rigid or rigidity may refer to:.
A streambed or stream bed is the bottom of a stream or river and is confined within a channel or the banks of the waterway. Usually, the bed does not contain terrestrial (land) vegetation and instead supports different types of aquatic vegetation, depending on the type of streambed material and water velocity. Streambeds are what would be left once a stream is no longer in existence. The beds are usually well preserved even if they get buried because the banks and canyons made by the stream are typically hard, although soft sand and debris often fill the bed. Dry, buried streambeds can actually be underground water pockets. During times of rain, sandy streambeds can soak up and retain water, even during dry seasons, keeping the water table close enough to the surface to be obtainable by local people.
A robot is a machine, especially one programmable via a computer, capable of automatically carrying out a complex series of actions. A robot can be guided by an external or internal control device. Robots may be humanoid, but most are task-performing machines prioritizing functionality over aesthetics.
Rotation, rotational or rotary motion is the movement of an object that leaves at least one point unchanged. In 2 dimensions, a plane figure can rotate in either a clockwise or counterclockwise sense around a point called the center of rotation. In 3 dimensions, a solid figure rotates around an imaginary line called an axis of rotation.
Runoff, run-off or RUNOFF may refer to:Runoff (hydrology), the flow of water over land
Channel runoff, the confined flow of water
Surface runoff, the unconfined flow of water over land
Runoff model (reservoir), a mathematical model involving rainfall and runoff
Runoff curve number, an empirical parameter used in hydrology
RUNOFF, the first computer text-formatting program
Runoff or run-off, another name for bleed, printing that lies beyond the edges to which a printed sheet is trimmed
Runoff or run-off, a stock market term
Runoff voting system, also known as the two-round system, a voting system where a second round of voting is used to elect one of the two candidates receiving the most votes in the first round
Instant-runoff voting, an extension or variation of runoff voting where a second round can be rendered unnecessary by voters ranking candidates in order of preference
Run-off area, a racetrack safety feature
Runoff directed by Kimberly Levin.
Sacrifice is an act or offering made to a deity. A sacrifice can serve as propitiation, or a sacrifice can be an offering of praise and thanksgiving.
Salinity is the saltiness or amount of salt dissolved in a body of water, called saline water. It is usually measured in g/L or g/kg.
Sampling may refer to:Sampling, converting a continuous signal into a discrete signal
Sampling (graphics), converting continuous colors into discrete color components
Sampling (music), the reuse of a sound recording in another recording
Sampler, an electronic musical instrument used to record and play back samples
Sampling (statistics), selection of observations to acquire some knowledge of a statistical population
Sampling, selection of cases for single or multiple case studies
Sampling (audit), application of audit procedures to less than 100% of population to be audited
Sampling (medicine), gathering of matter from the body to aid in the process of a medical diagnosis and/or evaluation of an indication for treatment, further medical tests or other procedures.
Sampling, detection of hazardous materials in the workplace
Sampling, taking a representative portion of a material or product to test, typically for the purposes of identification, quality control, or regulatory assessment. See Sample (material).
Sandstone is a clastic sedimentary rock composed mainly of sand-sized silicate grains, cemented together by another mineral. Sandstones comprise about 20–25% of all sedimentary rocks.
Sanitation refers to public health conditions related to clean drinking water and treatment and disposal of human excreta and sewage. Preventing human contact with feces is part of sanitation, as is hand washing with soap. Sanitation systems aim to protect human health by providing a clean environment that will stop the transmission of disease, especially through the fecal–oral route. For example, diarrhea, a main cause of malnutrition and stunted growth in children, can be reduced through adequate sanitation. There are many other diseases which are easily transmitted in communities that have low levels of sanitation, such as ascariasis, cholera, hepatitis, polio, schistosomiasis, and trachoma, to name just a few.
Saturation, saturated, unsaturation or unsaturated may refer to:.
Scaffolding, also called scaffold or staging, is a temporary structure used to support a work crew and materials to aid in the construction, maintenance and repair of buildings, bridges and all other human-made structures. Scaffolds are widely used on site to get access to heights and areas that would be otherwise hard to get to. Unsafe scaffolding has the potential to result in death or serious injury. Scaffolding is also used in adapted forms for formwork and shoring, grandstand seating, concert stages, access/viewing towers, exhibition stands, ski ramps, half pipes and art projects.
In physics, scattering is a wide range of physical processes where moving particles or radiation of some form, such as light or sound, are forced to deviate from a straight trajectory by localized non-uniformities in the medium through which they pass. In conventional use, this also includes deviation of reflected radiation from the angle predicted by the law of reflection. Reflections of radiation that undergo scattering are often called diffuse reflections and unscattered reflections are called specular (mirror-like) reflections. Originally, the term was confined to light scattering. As more "ray"-like phenomena were discovered, the idea of scattering was extended to them, so that William Herschel could refer to the scattering of "heat rays" in 1800. John Tyndall, a pioneer in light scattering research, noted the connection between light scattering and acoustic scattering in the 1870s. Near the end of the 19th century, the scattering of cathode rays and X-rays was observed and discussed. With the discovery of subatomic particles and the development of quantum theory in the 20th century, the sense of the term became broader as it was recognized that the same mathematical frameworks used in light scattering could be applied to many other phenomena.
A schism is a division between people, usually belonging to an organization, movement, or religious denomination. The word is most frequently applied to a split in what had previously been a single religious body, such as the Great East–West Schism or the Western Schism. It is also used of a split within a non-religious organization or movement or, more broadly, of a separation between two or more people, be it brothers, friends, lovers, etc.
Scholasticism was a medieval European philosophical movement or methodology that was the predominant education in Europe from about 1100 to 1700. It is known for employing logically precise analyses toward the goal of reconciling classical philosophy and Catholic Christianity.
The seabed is the bottom of the ocean. Alternatively, it is known as the seafloor, sea floor, ocean floor, or ocean bottom. The floor of all the world's oceans is known as the seabed.
Seagrasses are the only flowering plants which grow in marine environments. There are about 60 species of fully marine seagrasses which belong to four families, all in the order Alismatales. Seagrasses evolved from terrestrial plants which recolonised the ocean 70 to 100 million years ago.
A seamount is a large submarine landform that rises from the ocean floor without reaching the water surface, and thus is not an island, islet, or cliff-rock. Seamounts are typically formed from extinct volcanoes that rise abruptly and are usually found rising from the seafloor to 100–4,000 m (330–13,120 ft) in height. They are defined by oceanographers as independent features that rise to at least 1,000 m (3,281 ft) above the seafloor, characteristically of conical form. The peaks are often found hundreds to thousands of meters below the surface, and are therefore considered to be within the deep sea. During their evolution over geologic time, the largest seamounts may reach the sea surface where wave action erodes the summit to form a flat surface. After they have subsided and sunk below the sea surface, such flat-top seamounts are called "guyots" or "tablemounts".
Secretion is the movement of material from one point to another, such as a secreted chemical substance from a cell or gland. In contrast, excretion is the removal of certain substances or waste products from a cell or organism. The classical mechanism of cell secretion is via secretory portals at the plasma membrane called porosomes. Porosomes are permanent cup-shaped lipoprotein structures embedded in the cell membrane, where secretory vesicles transiently dock and fuse to release intra-vesicular contents from the cell.
Sediment is a solid material made of loose particles that is transported to a new location where it is deposited. It occurs naturally and, through the processes of weathering and erosion, is broken down and subsequently transported by the action of wind, water, or ice or by the force of gravity acting on the particles. For example, sand and silt can be carried in suspension in river water and on reaching the sea bed deposited by sedimentation; if buried, they may eventually become sandstone and siltstone through lithification.
Seismicity is a measure encompassing earthquake occurrences, mechanisms, and magnitude at a given geographical location. As such, it summarizes a region's seismic activity. The term was coined by Beno Gutenberg and Charles Francis Richter in 1941. Seismicity is studied by geophysicists.
Semantics is the study of linguistic meaning. It examines what meaning is, how words get their meaning, and how the meaning of a complex expression depends on its parts. Part of this process involves the distinction between sense and reference. Sense is given by the ideas and concepts associated with an expression while reference is the object to which an expression points. Semantics contrasts with syntax, which studies the rules that dictate how to create grammatically correct sentences, and pragmatics, which investigates how people use language in communication. Semantics, together with syntactics and pragmatics, is a part of semiotics.
Senescence or biological aging is the gradual deterioration of functional characteristics in living organisms. Whole organism senescence involves an increase in death rates or a decrease in fecundity with increasing age, at least in the later part of an organism's life cycle. However, the effects of senescence can be delayed. The 1934 discovery that calorie restriction can extend lifespans by 50% in rats, the existence of species having negligible senescence, and the existence of potentially immortal organisms such as members of the genus Hydra have motivated research into delaying senescence and thus age-related diseases. Rare human mutations can cause accelerated aging diseases.
Sensory may refer to:.
Separation may refer to:.
Serology is the scientific study of serum and other body fluids. In practice, the term usually refers to the diagnostic identification of antibodies in the serum. Such antibodies are typically formed in response to an infection, against other foreign proteins, or to one's own proteins.
Settlement may refer to:Human settlement, a community where people live
Settlement (structural), downward movement of a structure's foundation
Settlement (finance), where securities are delivered against payment of money
Settlement (litigation), a resolution between disputing parties about a legal case
Settlement (trust), a deed whereby property is given by a settlor into trust
Thomson Bay Settlement, Rottnest Island, Western Australia, also known as simply The Settlement
Closing, the final step in executing a real estate transaction.
In computer graphics, a shader is a programmable operation which is applied to data as it moves through the rendering pipeline. Shaders can act on data such as vertices and primitives—to generate or morph geometry—and fragments –to calculate the values in a rendered image.
Shear may refer to:.
Shellfish, in colloquial and fisheries usage, are exoskeleton-bearing aquatic invertebrates used as food, including various species of molluscs, crustaceans, and echinoderms. Although most kinds of shellfish are harvested from saltwater environments, some are found in freshwater. In addition, a few species of land crabs are eaten, for example Cardisoma guanhumi in the Caribbean. Shellfish are among the most common food allergens.
A signal is both the process and the result of transmission of data over some media accomplished by embedding some variation. Signals are important in multiple subject fields, including signal processing, information theory and biology.
A silicate is any member of a family of polyatomic anions consisting of silicon and oxygen, usually with the general formula [SiO(4−2x)−4−x]n, where 0 ≤ x < 2. The family includes orthosilicate SiO4−4, metasilicate SiO2−3, and pyrosilicate Si2O6−7. The name is also used for any salt of such anions, such as sodium metasilicate; or any ester containing the corresponding chemical group, such as tetramethyl orthosilicate. The name "silicate" is sometimes extended to any anions containing silicon, even if they do not fit the general formula or contain other atoms besides oxygen; such as hexafluorosilicate [SiF6]2−. Most commonly, silicates are encountered as silicate minerals.
A simulation is an imitative representation of a process or system that could exist in the real world. In this broad sense, simulation can often be used interchangeably with model. Sometimes a clear distinction between the two terms is made, in which simulations require the use of models; the model represents the key characteristics or behaviors of the selected system or process, whereas the simulation represents the evolution of the model over time. Another way to distinguish between the terms is to define simulation as experimentation with the help of a model. This definition includes time-independent simulations. Often, computers are used to execute the simulation.
Sintering or frittage is the process of compacting and forming a solid mass of material by pressure or heat without melting it to the point of liquefaction. Sintering happens as part of a manufacturing process used with metals, ceramics, plastics, and other materials. The atoms/molecules in the sintered material diffuse across the boundaries of the particles, fusing the particles together and creating a solid piece.
Skepticism (US) or scepticism (UK) is a questioning attitude or doubt toward knowledge claims that are seen as mere belief or dogma. For example, if a person is skeptical about claims made by their government about an ongoing war then the person doubts that these claims are accurate. In such cases, skeptics normally recommend not disbelief but suspension of belief, i.e. maintaining a neutral attitude that neither affirms nor denies the claim. This attitude is often motivated by the impression that the available evidence is insufficient to support the claim. Formally, skepticism is a topic of interest in philosophy, particularly epistemology.
A skylight is a light-permitting structure or window, usually made of transparent or translucent glass, that forms all or part of the roof space of a building for daylighting and ventilation purposes.
Slippage may refer to:Degree of slipping or loosening as result of slipperiness
Slippage (finance), the difference between estimated transaction costs and the amount actually paid
Project slippage, in project planning, the act of missing a deadline
Replication slippage, nucleotide duplications created by DNA polymerase during DNA replication
Bit slip, the loss or gain of a bit or bits, caused by variations in respective clock rates of transmitting and receiving devices.
Snowpack is an accumulation of snow that compresses with time and melts seasonally, often at high elevation or high latitude. Snowpacks are an important water resource that feed streams and rivers as they melt, sometimes leading to flooding. Snowpacks provide water to down-slope communities for drinking and agriculture. High-latitude or high-elevation snowpacks contribute mass to glaciers in their accumulation zones, where annual snow deposition exceeds annual melting.
Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method.
Could not find summary for "Solarwind".
A solstice is the time when the Sun reaches its most northerly or southerly excursion relative to the celestial equator on the celestial sphere. Two solstices occur annually, around 20–22 June and 20–22 December. In many countries, the seasons of the year are defined by reference to the solstices and the equinoxes.
A solvent is a substance that dissolves a solute, resulting in a solution. A solvent is usually a liquid but can also be a solid, a gas, or a supercritical fluid. Water is a solvent for polar molecules, and the most common solvent used by living things; all the ions and proteins in a cell are dissolved in water within the cell.
Sonar is a technique that uses sound propagation to navigate, measure distances (ranging), communicate with or detect objects on or under the surface of the water, such as other vessels.
Sorption is a physical and chemical process by which one substance becomes attached to another. Specific cases of sorption are treated in the following articles:Absorption"the incorporation of a substance in one state into another of a different state" ;
AdsorptionThe physical adherence or bonding of ions and molecules onto the surface of another phase ;
Ion exchangeAn exchange of ions between two electrolytes or between an electrolyte solution and a complex.
A soundscape is the acoustic environment as perceived by humans, in context. The term, originally coined by Michael Southworth, was popularized by R. Murray Schafer. There is a varied history of the use of soundscape depending on discipline, ranging from urban design to wildlife ecology to computer science. An important distinction is to separate soundscape from the broader acoustic environment. The acoustic environment is the combination of all the acoustic resources, natural and artificial, within a given area as modified by the environment. The International Organization for Standardization (ISO) standardized these definitions in 2014.
Sovereignty is generally defined as supreme, independent control and lawmaking authority over a territory. It is expressed through the power to rule and make law. Sovereignty entails hierarchy within a state as well as external autonomy, which refers to the ability of a state to act independently in international affairs. In any state, sovereignty is assigned to the person, body or institution that has the ultimate authority over its citizens and the power to modify existing laws. In political theory, sovereignty is a substantive term designating supreme legitimate authority over some polity. According to international law, sovereign states are all considered equal, and no state has the right to interfere in the internal affairs of another sovereign state. While Article 2(7) of the UN Charter explicitly recognizes the sovereignty of states, and in general there is a principle of non-interference in the domestic affairs of sovereign states, the UN Security Council’s Chapter VII powers clearly contemplate the use of force against a state when necessary to restore peace. Moreover, the recent Responsibility to Protect (R2P) authorizes the United Nations to take action to “avert a humanitarian catastrophe” within a state when that state's government cannot or will not act.
Spectroscopy is the field of study that measures and interprets electromagnetic spectra as it interacts with matter. In narrower contexts, spectroscopy is the precise study of color as generalized from radiated visible light to all bands of the electromagnetic spectrum.
A spectrum is a set of related ideas, objects, or properties whose features overlap such that they blend to form a continuum. The word spectrum was first used scientifically in optics to describe the rainbow of colors in visible light after passing through a prism. In the optical spectrum, light wavelength is viewed as continuous, and spectral colors are seen to blend into one another smoothly when organized in order of their corresponding wavelengths. As scientific understanding of light advanced, the term came to apply to the entire electromagnetic spectrum, including radiation not visible to the human eye.
In finance, speculation is the purchase of an asset with the hope that that asset will become more valuable in a brief amount of time.
The term can also refer to short sales, in which the speculator hopes for a decline in value. Speculation often has a pejorative connotation, as the activity is linked to bubbles, economic downturns, and financial crises.
In mathematics, a spiral is a curve which emanates from a point, moving further away as it revolves around the point. It is a subtype of whorled patterns, a broad group that also includes concentric objects.
In biology, a spore is a unit of sexual or asexual reproduction that may be adapted for dispersal and for survival, often for extended periods of time, in unfavourable conditions. Spores form part of the life cycles of many plants, algae, fungi and protozoa. They were thought to have appeared as early as the mid-late Ordovician period as an adaptation of early land plants.
Stability may refer to:.
A stalagmite
is a type of rock formation that rises from the floor of a cave due to the accumulation of material deposited on the floor from ceiling drippings. Stalagmites are typically composed of calcium carbonate, but may consist of lava, mud, peat, pitch, sand, sinter, and amberat.
A stalactite is a mineral formation that hangs from the ceiling of caves, hot springs, or man-made structures such as bridges and mines. Any material that is soluble and that can be deposited as a colloid, or is in suspension, or is capable of being melted, may form a stalactite. Stalactites may be composed of lava, minerals, mud, peat, pitch, sand, sinter, and amberat. A stalactite is not necessarily a speleothem, though speleothems are the most common form of stalactite because of the abundance of limestone caves.
Standardization or standardisation is the process of implementing and developing technical standards based on the consensus of different parties that include firms, users, interest groups, standards organizations and governments. Standardization can help maximize compatibility, interoperability, safety, repeatability, efficiency, and quality. It can also facilitate a normalization of formerly custom processes.
Starburst most often refers to:Starburst region, a generic term to describe a region of space with a much higher than normal star formation
Starburst galaxy, a galaxy with an exceptionally high rate of star formation
Starburst (candy), a brand of fruit-flavored candy.
A state is a political entity that regulates society and the population within a definite territory. Government is considered to form the fundamental apparatus of contemporary states.
In mathematics and statistics, a stationary process is a stochastic process whose statistical properties, such as mean and variance, do not change over time. More formally, the joint probability distribution of the process remains the same when shifted in time. This implies that the process is statistically consistent across different time periods. Because many statistical procedures in time series analysis assume stationarity, non-stationary data are frequently transformed to achieve stationarity before analysis.
In physical geography, a steppe is an ecoregion characterized by grassland plains without closed forests except near rivers and lakes.
Sterile or sterility may refer to:Asepsis, a state of being free from biological contaminants
Sterile (archaeology), a sediment deposit which contains no evidence of human activity
Sterilization (microbiology), any process that eliminates or kills all forms of life or removes them from an item or a field
Sterility (physiology), an inability of a living organism to effect sexual reproduction
Infertility, a medical condition which prevents a person, an animal or a plant from bearing children, especially through natural means
Sterile Records, a record label which was formed by Nigel Ayers and Caroline K of the post-industrial music group Nocturnal Emissions in London in 1979.
Stimulants are a class of psychoactive drugs that increase alertness. They are used for various purposes, such as enhancing attention, motivation, cognition, mood, and physical performance. Some stimulants occur naturally, while others are exclusively synthetic. Common stimulants include caffeine, nicotine, cocaine, amphetamine/methamphetamine, methylphenidate, and modafinil. Stimulants may be subject to varying forms of regulation, or outright prohibition, depending on jurisdiction. Most stimulants are highly addictive and damage health when addicted.
Stoichiometry is the relationships between the quantities of reactants and products before, during and after chemical reactions.
Stratification may refer to:.
The stratosphere is the second-lowest layer of the atmosphere of Earth, located above the troposphere and below the mesosphere. Pronounced, the name originates from Ancient Greek  στρωτός (strōtós) 'layer, stratum' and  -sphere. The stratosphere is composed of stratified temperature zones, with the warmer layers of air located higher and the cooler layers lower. The increase of temperature with altitude is a result of the absorption of the Sun's ultraviolet (UV) radiation by the ozone layer, where ozone is exothermically photolyzed into oxygen in a cyclical fashion. This temperature inversion is in contrast to the troposphere, where temperature decreases with altitude, and between the troposphere and stratosphere is the tropopause border that demarcates the beginning of the temperature inversion.
Stress may refer to:.
Subduction is a geological process in which the oceanic lithosphere and some continental lithosphere is recycled into the Earth's mantle at the convergent boundaries between tectonic plates. Where one tectonic plate converges with a second plate, the heavier plate dives beneath the other and sinks into the mantle. A region where this process occurs is known as a subduction zone, and its surface expression is known as an arc-trench complex. The process of subduction has created most of the Earth's continental crust. Rates of subduction are typically measured in centimeters per year, with rates of convergence as high as 11 cm/year.
Substrate may refer to:.
Succession is the act or process of following in order or sequence.
A supernova is a powerful and luminous explosion of a star. A supernova occurs during the last evolutionary stages of a massive star, or when a white dwarf is triggered into runaway nuclear fusion. The original object, called the progenitor, either collapses to a neutron star or black hole, or is completely destroyed to form a diffuse nebula. The peak optical luminosity of a supernova can be comparable to that of an entire galaxy before fading over several weeks or months.
Supremacy may refer to:.
A surface, as the term is most generally used, is the outermost or uppermost layer of a physical object. It is the portion or region of the object that can first be observed and with which other objects first interact.
Surge means a sudden transient rush or flood, and may refer to:.
Survival or survivorship, the act of surviving, is the propensity of something to continue existing despite conditions that might kill or destroy it. The concept can be applied to humans and other living things, to a physical object, and to abstract things such as beliefs or ideas. Living things generally have a self-preservation instinct to survive, while objects intended for use in harsh conditions are designed for survivability.
Symbiosis is any close and long-term biological interaction between two organisms of different species. In 1879, Heinrich Anton de Bary defined symbiosis as "the living together of unlike organisms". The term is sometimes more exclusively used in a restricted, mutualistic sense, where both symbionts contribute to each other's subsistence. This means that they benefit each other in some way.
Symmetry in everyday life refers to a sense of harmonious and beautiful proportion and balance. In mathematics, the term has a more precise definition and is usually used to refer to an object that is invariant under some transformations, such as translation, reflection, rotation, or scaling. Although these two meanings of the word can sometimes be told apart, they are intricately related, and hence are discussed together in this article.
Synthesis or synthesize may refer to:.
In classical Greek mythology, Syrinx was an Arcadian nymph and a follower of Artemis, known for her chastity. Being pursued by Pan, she fled into the river Ladon, and at her own request was metamorphosed into a reed from which Pan then made his panpipes.
Taxonomy is a practice and science concerned with classification or categorization. Typically, there are two parts to it: the development of an underlying scheme of classes and the allocation of things to the classes (classification).
Technocracy is an expert-based type of governance. In its strongest sense, it is a form of government in which decisions across all sectors and policy domains follow evidence-based, efficiency-oriented procedures grounded in scientific methods and instrumental rationality. In a weaker sense, the term denotes hybrid models that delegate specific functions to experts or implement expertise-driven decision procedures in areas such as central banking, public health, or environmental regulation.
Telemetry is the in situ collection of measurements or other data at remote points and their automatic transmission to receiving equipment (telecommunication) for monitoring. The word is derived from the Greek roots tele, 'far off', and metron, 'measure'. Systems that need external instructions and data to operate require the counterpart of telemetry: telecommand.
In psychology, temperament broadly refers to consistent individual differences in behavior that are biologically based and are relatively independent of learning, system of values and attitudes.
Temperature is a numerical expression of hotness or coldness. Temperature is measured with a thermometer. It reflects the average kinetic energy of the vibrating and colliding atoms making up a substance.
Tension may refer to:.
Ternary or trinary is an adjective meaning "composed of three items". It can refer to:.
A territory is an area of land, sea, or space, belonging or connected to a particular country, person, or animal.
A tessellation or tiling is the covering of a surface, often a plane, using one or more geometric shapes, called tiles, with no overlaps and no gaps. In mathematics, tessellation can be generalized to higher dimensions and a variety of geometries.
A thermocline is
a distinct layer based on temperature within a large body of fluid with a high gradient of distinct temperature differences associated with depth. In the ocean, the thermocline divides the upper mixed layer from the calm deep water below.
A thermometer is a device that measures temperature or temperature gradient. A thermometer has two important elements: (1) a temperature sensor in which some change occurs with a change in temperature; and (2) some means of converting this change into a numerical value. Thermometers are widely used in technology and industry to monitor processes, in meteorology, in medicine, and in scientific research.
Threshold may refer to:.
Tidal is the adjectival form of tide.
Titration is a common laboratory method of quantitative chemical analysis to determine the concentration of an identified analyte. A reagent, termed the titrant or titrator, is prepared as a standard solution of known concentration and volume. The titrant reacts with a solution of analyte to determine the analyte's concentration. The volume of titrant that reacted with the analyte is termed the titration volume.
Topology is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
A tornado, also known as a twister, is a rapidly rotating column of air that extends vertically from the surface of the Earth to the base of a cumulonimbus or cumulus cloud. Tornadoes are often visible in the form of a condensation funnel originating from the cloud base, with a cloud of rotating debris and dust close to the ground. Most tornadoes have wind speeds less than 180 kilometers per hour, are about 80 meters across, and travel several kilometers before dissipating. The most extreme tornadoes can attain wind speeds of more than 480 kilometers per hour (300 mph), can be more than 3 kilometers (2 mi) in diameter, and can stay on the ground for more than 100 km (62 mi).
Toxicity is the degree to which a chemical substance or a particular mixture of substances can damage an organism. Toxicity can refer to the effect on a whole organism, such as an animal, bacterium, or plant, as well as the effect on a substructure of the organism, such as a cell (cytotoxicity) or an organ such as the liver (hepatotoxicity). Sometimes the word is more or less synonymous with poisoning in everyday usage.
A trajectory is the path an object takes through its motion over time. In classical mechanics, a trajectory is defined by Hamiltonian mechanics via canonical coordinates; hence, a complete trajectory is defined by position and momentum, simultaneously.
Transcription refers to the process of converting sounds into letters or musical notes, or producing a copy of something in another medium, including:.
A transducer is a device that usefully converts energy from one form to another. Usually a transducer converts a signal in one form of energy to a signal in another.
Transducers are often employed at the boundaries of automation, measurement, and control systems, where electrical signals are converted to and from other physical quantities. The process of converting one form of energy to another is known as transduction.
Transfer may refer to:.
Transformation may refer to:.
A transistor is a semiconductor device used to amplify or switch electrical signals and power. It is one of the basic building blocks of modern electronics. It is composed of semiconductor material, usually with at least three terminals for connection to an electronic circuit. A voltage or current applied to one pair of the transistor's terminals controls the current through another pair of terminals. Because the controlled (output) power can be higher than the controlling (input) power, a transistor can amplify a signal. Some transistors are packaged individually, but many more in miniature form are found embedded in integrated circuits. Because transistors are the key active components in practically all modern electronics, many people consider them one of the 20th century's greatest inventions.
Transmutation may refer to:.
Transport or transportation is the intentional movement of humans, animals, and goods from one location to another. Modes of transport include air, land, water, cable, pipelines, and space. The field can be divided into infrastructure, vehicles, and operations. Transport enables human trade, which is essential for the development of civilizations.
A tremor is an involuntary, somewhat rhythmic muscle contraction and relaxation involving oscillations or twitching movements of one or more body parts. It is the most common of all involuntary movements and can affect the hands, arms, eyes, face, head, vocal folds, trunk, and legs. Most tremors occur in the hands. In some people, a tremor is a symptom of another neurological disorder.
A tributary, or an affluent, is a stream or river that flows into a larger stream, river, or a lake. A tributary does not flow directly into a sea or ocean. Tributaries, and the main stem river into which they flow, drain the surrounding drainage basin of its surface water and groundwater, leading the water out into an ocean, another river, or into an endorheic basin.
The tropics are the region of Earth surrounding the equator, where the Sun may shine directly overhead. This contrasts with the temperate or polar regions of Earth, where the Sun can never be directly overhead. Because of Earth's axial tilt, the width of the tropics is twice the tilt. The tropics are also referred to as the tropical zone and the torrid zone.
A tsunami is a series of waves in a water body caused by the displacement of a large volume of water, generally in an ocean or a large lake. Earthquakes, volcanic eruptions and underwater explosions above or below water all have the potential to generate a tsunami. Unlike normal ocean waves, which are generated by wind, or tides, which are in turn generated by the gravitational pull of the Moon and the Sun, a tsunami is generated by the displacement of water from a large event.
In fluid dynamics, turbulence or turbulent flow is fluid motion exhibiting chaotic changes in pressure and flow velocity. It is in contrast to laminar flow, which occurs when a fluid flows in parallel layers with no disruption between those layers.
Ultrasound is sound with frequencies greater than 20 kilohertz. This frequency is the approximate upper audible limit of human hearing in healthy young adults. The physical principles of acoustic waves apply to any frequency range, including ultrasound. Ultrasonic devices operate with frequencies from 20 kHz up to several gigahertz.
Ultraviolet radiation (UV) is electromagnetic radiation of wavelengths of 100–400 nanometers, shorter than that of visible light, but longer than X-rays. Wavelengths between 10 and 100 nanometers are called extreme ultraviolet and share some properties with soft X-rays. UV radiation is present in sunlight and constitutes about 10% of the total electromagnetic radiation output from the Sun. It is also produced by electric arcs, Cherenkov radiation, and specialized lights, such as mercury-vapor lamps, tanning lamps, and black lights.
Uncertainty or incertitude refers to situations involving imperfect or unknown information. It applies to predictions of future events, to physical measurements that are already made, or to the unknown, and is particularly relevant for decision-making. Uncertainty arises in partially observable or stochastic or complex or dynamic environments, as well as due to ignorance, indolence, or both. It arises in any number of fields, including insurance, philosophy, physics, statistics, economics, entrepreneurship, finance, medicine, psychology, sociology, engineering, metrology, meteorology, ecology and information science.
Undercurrent is a flow of water below the surface:In an ocean, a subsurface current, a water current which flows beneath and usually independently of surface currents.
In a river, a subsurface current .
Underground most commonly refers to:Subterranea (geography), the regions beneath the surface of the Earth.
Unification or unification theory may refer to:.
Uplift may refer to:.
Uranium is a chemical element; it has symbol U and atomic number 92. It is a silvery-grey metal in the actinide series of the periodic table. A uranium atom has 92 protons and 92 electrons, of which 6 are valence electrons. Uranium radioactively decays, usually by emitting an alpha particle. The half-life of this decay varies between 159,200 and 4.5 billion years for different isotopes, making them useful for dating the age of the Earth. The most common isotopes in natural uranium are uranium-238 and uranium-235. Uranium has the highest atomic weight of the primordially occurring elements. Its density is about 70% higher than that of lead and slightly lower than that of gold or tungsten. It occurs naturally in low concentrations of a few parts per million in soil, rock and water, and is commercially extracted from uranium-bearing minerals such as uraninite.
Urbanism is the scientific study of how inhabitants of urban areas, such as towns and cities, interact with the built environment. It is a direct component of disciplines such as urban planning, a profession focusing on the design and management of urban areas, and urban sociology, an academic field which studies urban life.
Vacancy or No Vacancy may refer to:.
A vaccine is a biological preparation that provides active acquired immunity to a particular infectious or malignant disease. The safety and effectiveness of vaccines has been widely studied and verified. A vaccine typically contains an agent that resembles a disease-causing microorganism and is often made from weakened or killed forms of the microbe, its toxins, or one of its surface proteins. The agent stimulates the immune system to recognize the agent as a threat, destroy it, and recognize further and destroy any of the microorganisms associated with that agent that it may encounter in the future.
A vacuole is a membrane-bound organelle which is present in plant and fungal cells and some protist, animal, and bacterial cells. Vacuoles are essentially enclosed compartments which are filled with water containing inorganic and organic molecules including enzymes in solution, though in certain cases they may contain solids which have been engulfed. Vacuoles are formed by the fusion of multiple membrane vesicles and are effectively just larger forms of these. The organelle has no basic shape or size; its structure varies according to the requirements of the cell.
Valence or valency may refer to:.
Vaporization of an element or compound is a phase transition from the liquid phase to vapor. There are two types of vaporization: evaporation and boiling. Evaporation is a surface phenomenon, whereas boiling is a bulk phenomenon.
Variable may refer to:.
In probability theory and statistics, variance is the expected value of the squared deviation from the mean of a random variable. The standard deviation is obtained as the square root of the variance. Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers are spread out from their average value. It is the second central moment of a distribution, and the covariance of the random variable with itself, and it is often represented by ⁠⁠, ⁠⁠, ⁠⁠, ⁠⁠, or ⁠⁠.
Vector most often refers to:Disease vector, an agent that carries and transmits an infectious pathogen into another living organism
Euclidean vector, a quantity with a magnitude and a direction.
Velocity is a measurement of speed in a certain direction of motion. It is a fundamental concept in kinematics, the branch of classical mechanics that describes the motion of physical objects. Velocity is a vector quantity, meaning that both magnitude and direction are needed to define it. The scalar absolute value (magnitude) of velocity is called speed, a quantity that is measured in metres per second in the SI (metric) system. For example, "5 metres per second" is a scalar, whereas "5 metres per second east" is a vector. If there is a change in speed, direction or both, then the object is said to be undergoing an acceleration.
Veneer may refer to:.
Ventilation may refer to:Ventilation (physiology), the movement of air between the environment and the lungs via inhalation and exhalation
Mechanical ventilation, in medicine, using artificial methods to assist breathing
Respirator, a machine designed to move breathable air into and out of the lungs
Ventilation (architecture), the process of "changing" or replacing air in any space to provide high indoor air quality
Ventilation (firefighting), the expulsion of heat and smoke from a fire building
Ventilation (mining), flow of air to the underground workings of a mine of sufficient volume to dilute and remove noxious gases.
Vertebrates, also called craniates, are animals with a vertebral column and a cranium. The vertebral column surrounds and protects the spinal cord, while the cranium protects the brain.
In mechanics, vibration is oscillatory motion about an equilibrium point. Vibration may be deterministic if the oscillations can be characterised precisely, or random if the oscillations can only be analysed statistically.
The word Viral means "relating to viruses".
Virulence is a pathogen's or microorganism's ability to cause damage to a host.
When two fluid layers move relative to each other, a friction force develops between them and the slower layer acts to slow down the faster layer. This internal resistance to flow is described by the fluid property called viscosity, which reflects the internal stickiness of the fluid. In liquids, viscosity arises from cohesive molecular forces, while in gases it results from molecular collisions. Except for the case of superfluidity, there is no fluid with zero viscosity, and thus all fluid flows involve viscous effects to some degree.
Volatility or volatile may refer to:.
A volcano is a vent or fissure in the crust of a planetary-mass object, such as Earth, that allows hot lava, volcanic ash, and gases to escape from a magma chamber below the surface.
Voltage, also known as (electrical) potential difference, electric pressure, or electric tension, is the difference in electric potential between two points. In a static electric field, it corresponds to the work needed per unit of charge to move a positive test charge from the first point to the second point. In the International System of Units (SI), the derived unit for voltage is the volt (V).
In fluid dynamics, a vortex is a region in a fluid in which the flow revolves around an axis line, which may be straight or curved. Vortices form in stirred fluids and may be observed in smoke rings, whirlpools in the wake of a boat, and in the winds surrounding a tropical cyclone, tornado, or dust devil.
In physics and mathematics, wavelength or spatial period of a wave or periodic function is the distance over which the wave's shape repeats. In other words, it is the distance between consecutive corresponding points of the same phase on the wave, such as two adjacent crests, troughs, or zero crossings. Wavelength is a characteristic of both traveling waves and standing waves, as well as other spatial wave patterns. The inverse of the wavelength is called the spatial frequency. Wavelength is commonly designated by the Greek letter lambda (λ). For a modulated wave, wavelength may refer to the carrier wavelength of the signal. The term wavelength may also apply to the repeating envelope of modulated waves or waves formed by interference of several sinusoids.
Weathering is the deterioration of rocks, soils and minerals through contact with water, atmospheric gases, sunlight, and biological organisms. It occurs in situ, and so is distinct from erosion, which involves the transport of rocks and minerals by agents such as water, ice, snow, wind, waves and gravity.
A wetland is a distinct semi-aquatic ecosystem whose groundcovers are flooded or saturated in water, either permanently, for years or decades, or only seasonally. Flooding results in oxygen-poor (anoxic) processes taking place, especially in the soils. Wetlands form a transitional zone between waterbodies and dry lands, and are different from other terrestrial or aquatic ecosystems due to their vegetation's roots having adapted to oxygen-poor waterlogged soils. They are considered among the most biologically diverse of all ecosystems, serving as habitats to a wide range of aquatic and semi-aquatic plants and animals, with often improved water quality due to plant removal of excess nutrients such as nitrates and phosphorus.
A whirlpool is a body of rotating water produced by opposing currents or a current running into an obstacle. Miniature whirlpools form when a bath or a sink is draining. More powerful ones formed in seas or oceans may be called maelstroms. Vortex is the proper term for a whirlpool that has a downdraft.
Windfall or Windfalls may refer to:.
A windmill is a machine operated by the force of wind acting on vanes or sails to mill grain (gristmills), pump water, generate electricity, or drive other machinery.
Windmills were used throughout the high medieval and early modern periods; the horizontal or panemone windmill first appeared in Persia during the 9th century, and the vertical windmill first appeared in northwestern Europe in the 12th century. Regarded as an icon of Dutch culture, there are approximately 1,000 windmills in the Netherlands today.
A woodland is, in the broad sense, land covered with woody plants, or in a narrow sense, synonymous with wood, a low-density forest forming open habitats with plenty of sunlight and limited shade. Some savannas may also be woodlands, such as savanna woodland, where trees and shrubs form a light canopy.
In macroeconomics, the workforce or labour force is the sum of people either working or looking for work :.
A xenolith is a rock fragment that becomes enveloped in a larger rock during the latter's development and solidification. In geology, the term xenolith is almost exclusively used to describe inclusions in igneous rock entrained during magma ascent, emplacement and eruption. Xenoliths may be engulfed along the margins of a magma chamber, torn loose from the walls of an erupting lava conduit or explosive diatreme or picked up along the base of a flowing body of lava on the Earth's surface. A xenocryst is an individual foreign crystal included within an igneous body. Examples of xenocrysts are quartz crystals in a silica-deficient lava and diamonds within kimberlite diatremes. Xenoliths can be non-uniform within individual locations, even in areas which are spatially limited, e.g. rhyolite-dominated lava of Niijima volcano (Japan) contains two types of gabbroic xenoliths which are of different origin - they were formed in different temperature and pressure conditions.
Xenon is a chemical element; it has symbol Xe and atomic number 54. It is a dense, colorless, odorless noble gas found in Earth's atmosphere in trace amounts. Although generally unreactive, it can undergo a few chemical reactions such as the formation of xenon hexafluoroplatinate, the first noble gas compound to be synthesized.
Yield may refer to:.
Zeolites are a group of several microporous, crystalline aluminosilicate minerals commonly used as commercial adsorbents and catalysts. They mainly consist of silicon, aluminium, and oxygen, and have the general formula Mn+1/n(AlO2)−(SiO2)x･yH2O where Mn+1/n is either a metal ion or H+.
In European tradition, a zephyr is a light wind or a west wind, named after Zephyrus, the Greek god or personification of the west wind.
Could not find summary for "Zonation".
Zoology is the scientific study of animals. Its studies include the structure, embryology, classification, habits, and distribution of all animals, both living and extinct, and how they interact with their ecosystems. Zoology is one of the primary branches of biology. The term is derived from Ancient Greek  ζῷον (zôion) 'animal' and  λόγος (lógos) 'study of'.
Abbot is an ecclesiastical title given to the head of an independent monastery for men in various Western Christian traditions. The name is derived from abba, the Aramaic form of the Hebrew ab, and means "father". The female equivalent is abbess.
An abdomen is the front part of the torso between the thorax (chest) and pelvis in humans and in other vertebrates. The area occupied by the abdomen is called the abdominal cavity. In arthropods, it is the posterior tagma of the body; it follows the thorax or cephalothorax.
Abjuration is the solemn repudiation, abandonment, or renunciation by or upon oath, often the renunciation of citizenship or some other right or privilege. The term comes from the Latin abjurare, "to forswear".
Abolitionism, or the abolitionist movement, is the political movement to end slavery and liberate enslaved individuals around the world. It gained momentum in the western world in the late 18th and 19th centuries.
An abscess is a collection of pus that has built up within the tissue of the body, usually caused by bacterial infection. Signs and symptoms of abscesses include redness, pain, warmth, and swelling. The swelling may feel fluid-filled when pressed. The area of redness often extends beyond the swelling. Carbuncles and boils are types of abscess that often involve hair follicles, with carbuncles being larger. A cyst is related to an abscess, but it contains a material other than pus, and a cyst has a clearly defined wall. Abscesses can also form internally on internal organs and after surgery.
Absorption may refer to:.
Abundance may refer to:.
Abyss may refer to:.
An acetate is a salt formed by the combination of acetic acid with a base. "Acetate" also describes the conjugate base or ion typically found in aqueous solution and written with the chemical formula C2H3O−2. The neutral molecules formed by the combination of the acetate ion and a positive ion are also commonly called "acetates". The simplest of these is hydrogen acetate with corresponding salts, esters, and the polyatomic anion CH3CO−2, or CH3COO−.
Acidification may refer to:Ocean acidification, decrease in the pH of the Earth's oceans
Freshwater acidification, atmospheric depositions and soil leaching of SOx and NOx
Soil acidification, buildup of hydrogen cations, which reduces the soil pH
Souring, a cooking technique

.
An acropolis was the settlement of an upper part of an ancient Greek city, especially a citadel, and frequently a hill with precipitous sides, mainly chosen for purposes of defense. The term is typically used to refer to the Acropolis of Athens, yet nearly every Greek city had an acropolis of its own. Acropolises were used as religious centers and places of worship, forts, and places in which the royal and high-status resided. Acropolises became the nuclei of large cities of classical ancient times, and served as important centers of a community. Some well-known acropolises have become the centers of tourism in the present day, and they are a rich source of archaeological information of ancient Greece, especially, the Acropolis of Athens.
Acuity may refer to:.
Additive may refer to:.
Adjudication is the legal process by which an arbiter or judge reviews evidence and argumentation, including legal reasoning set forth by opposing parties or litigants, to come to a decision which determines rights and obligations between the parties involved.
Admiralty most often refers to:Admiralty, Hong Kong
Admiralty, military department in command of the Royal Navy from 1707 to 1964
The rank of admiral
Admiralty law.
Adrenaline, also known as epinephrine and alternatively spelled adrenalin, is a hormone and medication which is involved in regulating visceral functions. It appears as a white microcrystalline granule. Adrenaline is normally produced by the adrenal glands and by a small number of neurons in the medulla oblongata. It plays an essential role in the fight-or-flight response by increasing blood flow to muscles, heart output by acting on the SA node, pupil dilation response, and blood sugar level. It does this by binding to alpha and beta receptors. It is found in many animals, including humans, and some single-celled organisms. It has also been isolated from the plant Scoparia dulcis found in Northern Vietnam.
Aerospace refers to the technology and industry involved with the atmosphere and outer space collectively. Aerospace activity is very diverse, with a multitude of commercial, industrial, and military applications. Aerospace engineering consists of aeronautics and astronautics. Aerospace organizations research, design, manufacture, operate, maintain, and repair both aircraft and spacecraft.
Aesthetics is the branch of philosophy that studies beauty, taste, and related phenomena. In a broad sense, it includes the philosophy of art, which examines the nature of art, artistic creativity, the meanings of artworks, and audience appreciation.
Afforestation is the establishment of a forest or stand of trees in an area where there was no recent tree cover. There are three types of afforestation: natural regeneration, agroforestry and tree plantations. In the context of climate change, afforestation can be helpful for climate change mitigation through the route of carbon sequestration. Afforestation can also improve the local climate through increased rainfall and by being a barrier against high winds. The additional trees can also prevent or reduce topsoil erosion, floods and landslides. Finally, additional trees can be a habitat for wildlife, and provide employment and wood products.
In seismology, an aftershock is a smaller earthquake that follows a larger earthquake, in the same area of the main shock, caused as the displaced crust adjusts to the effects of the main shock. Large earthquakes can have hundreds to thousands of instrumentally detectable aftershocks, which steadily decrease in magnitude and frequency according to a consistent pattern. In some earthquakes the main rupture happens in two or more steps, resulting in multiple main shocks. These are known as doublet earthquakes, and in general can be distinguished from aftershocks in having similar magnitudes and nearly identical seismic waveforms.
Agility or nimbleness is an ability to change the body's position quickly and requires the integration of isolated movement skills using a combination of balance, coordination, speed, reflexes, strength, and endurance. More specifically, it is dependent on these six skills:Balance – The ability to maintain equilibrium when stationary or moving through the coordinated actions of our sensory functions ;
Static balance – The ability to retain the center of mass above the base of support in a stationary position;
Dynamic balance – The ability to maintain balance with body movement; an equal distribution of weight;
Speed – The ability to move all or part of the body quickly;
Strength – The ability of a muscle or muscle group to overcome a resistance; and lastly,
Coordination – The ability to control the movement of the body in co-operation with the body's sensory functions.
An agonist is a chemical that activates a receptor to produce a biological response. Receptors are cellular proteins whose activation causes the cell to modify what it is currently doing. In contrast, an antagonist blocks the action of the agonist, while an inverse agonist causes an action opposite to that of the agonist.
Agrarian means pertaining to agriculture, farmland, or rural areas.
An airship, dirigible balloon or dirigible is a type of aerostat (lighter-than-air) aircraft that can navigate through the air flying under its own power. Aerostats use buoyancy from a lifting gas that is less dense than the surrounding air to achieve the lift needed to stay airborne.
Albedo is the fraction of sunlight that is diffusely reflected by a body. It is measured on a scale from 0 to 1. Surface albedo is defined as the ratio of radiosity Je to the irradiance Ee received by a surface. The proportion reflected is not only determined by properties of the surface itself, but also by the spectral and angular distribution of solar radiation reaching the Earth's surface. These factors vary with atmospheric composition, geographic location, and time.
Alchemy is an ancient branch of natural philosophy, a philosophical and protoscientific tradition that was historically practised in China, India, the Muslim world, and Europe. In its Western form, alchemy is first attested in a number of pseudepigraphical texts written in Greco-Roman Egypt during the first few centuries AD. Greek-speaking alchemists often referred to their craft as "the Art" (τέχνη) or "Knowledge" (ἐπιστήμη), and it was often characterised as mystic (μυστική), sacred (ἱɛρά), or divine (θɛíα).
Algae is an informal umbrella term for any organisms from a large and diverse group of photosynthetic autotrophs/mixotrophs that are not plants (embryophytes), and includes species from numerous distinctly unrelated clades. Such organisms range from microscopic unicellular microalgae, which include cyanobacteria and phytoplankton such as Chlorella, Porphyridium and diatoms; to multicellular macroalgae such as ulvophytes, nori and kelp, which may grow up to 50 metres (160 ft) in length. Most algae are aquatic organisms, some in cohesive colonies, and macroscopic marine algae with complex, multicellular colonial structures are called seaweeds. The most complex freshwater algae are the Charophyta, a division of green algae which includes Spirogyra and stoneworts, the latter of which morphologically resemble aquatic grass. Most algae are planktons carried passively by water, although some macroalgae have developed holdfast structures that provide sessile anchorage. Algae exhibit a wide range of reproductive strategies, from simple asexual mitotic proliferation to complex forms of sexual reproduction via spores.
Algorithmic may refer to:Algorithm, step-by-step instructions for a calculation
Algorithmic art, art made by an algorithm
Algorithmic composition, music made by an algorithm
Algorithmic trading, trading decisions made by an algorithm
Algorithmic patent, an intellectual property right in an algorithm
Algorithmics, the science of algorithms
Algorithmica, an academic journal for algorithm research
Algorithmic efficiency, the computational resources used by an algorithm
Algorithmic information theory, study of relationships between computation and information
Algorithmic mechanism design, the design of economic systems from an algorithmic point of view
Algorithmic number theory, algorithms for number-theoretic computation
Algorithmic game theory, game-theoretic techniques for algorithm design and analysis
Algorithmic cooling, a phenomenon in quantum computation
Algorithmic probability, a universal choice of prior probabilities in Solomonoff's theory of inductive inference.
As a literary device or artistic form, an allegory is a narrative or visual representation in which a character, place, or event can be interpreted to represent a meaning with moral or political significance. Authors have used allegory throughout history in all forms of art to illustrate or convey complex ideas and concepts in ways that are comprehensible or striking to its viewers, readers, or listeners.
An alliance is a relationship among people, groups, or states that have joined together for mutual benefit or to achieve some common purpose, whether an explicit agreement has been worked out among them. Members of an alliance are called allies. Alliances form in many settings, including political alliances, military alliances, and business alliances.
Alluvium is loose clay, silt, sand, or gravel that has been deposited by running water in a stream bed, on a floodplain, in an alluvial fan or beach, or in similar settings. Alluvium is also sometimes called alluvial deposit. Alluvium is typically geologically young and is not consolidated into solid rock. Sediments deposited underwater, in seas, estuaries, lakes, or ponds, are not described as alluvium.
Floodplain alluvium can be highly fertile, and supported some of the earliest human civilizations.
An almanac is a regularly published listing of a set of current information about one or multiple subjects. It includes information like weather forecasts, farmers' planting dates, tide tables, and other tabular data often arranged according to the calendar. Celestial figures and various statistics are found in almanacs, such as the rising and setting times of the Sun and Moon, dates of eclipses, hours of high and low tides, and religious festivals. The set of events noted in an almanac may be tailored for a specific group of readers, such as farmers, sailors, or astronomers.
Altruism is concern for the well-being, the life, of others independently of personal benefit or reciprocity.
Amalgam most commonly refers to:Amalgam (chemistry), mercury alloy
Amalgam (dentistry), material of silver tooth fillings
Bonded amalgam, used in dentistry.
An ambassador is an official envoy, especially a high-ranking diplomat who represents a state and is usually accredited to another sovereign state or to an international organization as the resident representative of their own government or sovereign; or appointed for a special and often temporary diplomatic assignment. The word is also used informally for people who are known, without national appointment, to represent certain professions, activities, and fields of endeavor, such as sales.
Ambiguity is a state in which the meaning of a phrase, statement, situation, or resolution is not explicitly defined, making for several plausible interpretations. It arises when available information lacks sufficient context or a shared frame, so people cannot reliably determine what the problem is, what matters, what causes what, or what solution would count as correct. As a result, interpretation depends heavily on prior experience, assumptions, and imagination.
Ammonoids are extinct, typically coiled-shelled cephalopods composing the subclass Ammonoidea. They are more closely related to living octopuses, squid, and cuttlefish than they are to nautiluses, which they resemble. The earliest ammonoids appeared during the Emsian stage of the Early Devonian, around 410-408 million years ago, with the last species vanishing during or soon after the Cretaceous–Paleogene extinction event approximately 66 million years ago. They are often called ammonites, which is most frequently used for members of the order Ammonitida, the only remaining group of ammonoids from the Jurassic up until their extinction.
Amnesty is defined as "A pardon extended by a government to a group or class of people, usually for a political offense; the act of a sovereign power officially forgiving certain classes of people who are subject to trial but have not yet been convicted." Though the term general pardon has a similar definition, an amnesty constitutes more than a pardon, in so much as it obliterates all legal remembrance of the offense. Amnesty is increasingly used to express the idea of "freedom" and to refer to when prisoners can go free.
An amphitheatre is an open-air venue used for entertainment, performances, and sports. The term derives from the ancient Greek ἀμφιθέατρον, from ἀμφί, meaning "on both sides" or "around" and θέατρον, meaning "place for viewing".
The amplitude of a periodic variable is a measure of its change in a single period. The amplitude of a non-periodic signal is its magnitude compared with a reference value. There are various definitions of amplitude, which are all functions of the magnitude of the differences between the variable's extreme values. In older texts, the phase of a periodic function is sometimes called the amplitude.

Anarchy is a form of society without rulers. As a type of stateless society, it is commonly contrasted with states, which are polities that claim a monopoly on violence over a permanent territory. Beyond a lack of government, it can more precisely refer to societies that lack any form of authority or hierarchy. While viewed positively by anarchists, the primary advocates of anarchy, it is viewed negatively by advocates of statism, who see it in terms of social disorder.
Anemia is a blood disorder in which the blood has a reduced ability to carry oxygen. This can be due to a lower than normal number of red blood cells, a reduction in the amount of hemoglobin available for oxygen transport, or abnormalities in hemoglobin that impair its function. The name is derived from Ancient Greek  ἀν- (an-) 'not' and  αἷμα (haima) 'blood'.
Flowering plants are plants that bear flowers and fruits, and form the clade Angiospermae. The term angiosperm is derived from the Greek words ἀγγεῖον and σπέρμα, meaning that the seeds are enclosed within a fruit. The group was formerly called Magnoliophyta.
Anguish is "extreme unhappiness caused by physical or mental suffering." The feeling of anguish is typically preceded by a tragedy or event that has a profound meaning to the being in question. Anguish can be felt physically or mentally.
Anhydrite, or anhydrous calcium sulfate, is a mineral with the chemical formula CaSO4. It is in the orthorhombic crystal system, with three directions of perfect cleavage parallel to the three planes of symmetry. It is not isomorphous with the orthorhombic barium (baryte) and strontium (celestine) sulfates, as might be expected from the chemical formulas. Distinctly developed crystals are somewhat rare, the mineral usually presenting the form of cleavage masses. The Mohs hardness is 3.5, and the specific gravity is 2.9. The color is white, sometimes greyish, bluish, or purple. On the best developed of the three cleavages, the lustre is pearly; on other surfaces it is glassy. When exposed to water, anhydrite readily transforms to the more commonly occurring gypsum, (CaSO4·2H2O) by the absorption of water. This transformation is reversible, with gypsum or calcium sulfate hemihydrate forming anhydrite by heating to around 200 °C (400 °F) under normal atmospheric conditions. Anhydrite is commonly associated with calcite, halite, and sulfides such as galena, chalcopyrite, molybdenite, and pyrite in vein deposits.
In particle physics, annihilation is the process that occurs when a subatomic particle collides with its respective antiparticle to produce other particles, such as an electron colliding with a positron to produce two photons. The total energy and momentum of the initial pair are conserved in the process and distributed among a set of other particles in the final state. Antiparticles have exactly opposite additive quantum numbers from particles, so the sums of all quantum numbers of such an original pair are zero. Hence, any set of particles may be produced whose total quantum numbers are also zero as long as conservation of energy, conservation of momentum, and conservation of spin are obeyed.
Annulus or annular indicates a ring- or donut-shaped area or structure. It may refer to:.
Anomaly, The Anomaly or Anomalies may refer to:.
Anorexia nervosa (AN), often referred to simply as anorexia, is an eating disorder characterized by predominant food restriction, body image disturbance, fear of gaining weight, and an overwhelming desire to be thin. These characteristics often means individuals undergo severe malnutrition as a result of the disorder.
The Antarctic is the polar region of Earth that surrounds the South Pole, lying within the Antarctic Circle. It is diametrically opposite of the Arctic region around the North Pole.
An antibody (Ab), or immunoglobulin (Ig), is a large protein belonging to the immunoglobulin superfamily which is used by the immune system to identify and neutralize antigens such as those that exist on bacteria and virus cells, including those that cause disease. Each individual antibody recognizes one or more specific antigens, and antigens of virtually any size and chemical composition can be recognized. Each of the branching chains comprising the "Y" of an antibody contains a paratope that specifically binds to one particular epitope on an antigen, allowing the two molecules to bind together with precision. Using this mechanism, antibodies can effectively "tag" the antigen for attack by cells of the immune system, or can neutralize it directly.
Antiquity or Antiquities may refer to:.
The aorta is the main and largest artery in the human body, originating from the left ventricle of the heart, branching upwards immediately after, and extending down to the abdomen, where it splits at the aortic bifurcation into two smaller arteries. The aorta distributes oxygenated blood to all parts of the body through the systemic circulation.
Aphasia, also known as dysphasia, is an impairment in a person's ability to comprehend or formulate language because of dysfunction in specific brain regions. The major causes are stroke and head trauma; prevalence is hard to determine, but aphasia due to stroke is estimated to be 0.1–0.4% in developed countries. Aphasia can also be the result of brain tumors, epilepsy, autoimmune neurological diseases like multiple sclerosis, infection of the brain, or neurodegenerative diseases like dementias.
Aphids are small sap-sucking insects in the family Aphididae. Common names include greenfly and blackfly, although individuals within a species can vary widely in color. The group includes the fluffy white woolly aphids. A typical life cycle involves flightless females giving live birth to female nymphs—who may also be already pregnant, an adaptation scientists call telescoping generations—without the involvement of males. Maturing rapidly, females breed profusely so that the number of these insects multiplies quickly. Winged females may develop later in the season, allowing the insects to colonize new plants. In temperate regions, a phase of sexual reproduction occurs in the autumn, with the insects often overwintering as eggs.
An apsis is the farthest or nearest point in the orbit of a planetary body about its primary body. The line of apsides is the line connecting the two extreme values.
Apothecary is an archaic English term for a medical professional who formulates and dispenses materia medicacode: lat promoted to code: la  ('medicine') to physicians, surgeons and patients. The modern terms pharmacist and, in British English, chemist have taken over this role.
Apparatus may refer to:Technical term for a body of the Soviet and post-Soviet governments
Machine
Equipment
Critical apparatus, the critical and primary source material that accompanies an edition of a text
"Apparatus" (song), a song by Bombus
"Apparatus", a song by Deadguy from the album Fixation on a Co-Worker, 1995
Apparatus (band), an electro-industrial group active during the nineties
Apparatus (album), 1995 release by the band Apparatus
Apparatus (journal), an academic journal on film
In gymnastics, any of the individual events, or the equipment used in performing the event
A piece of laboratory equipment
in anatomy, a group of organs, see Apparatus (anatomy).
Appeasement, in an international context, is a diplomatic negotiation policy of making political, material, or territorial concessions to an aggressive power with intention to avoid conflict. The term is most often applied to the foreign policy between 1935 and 1939 of the British governments of Prime Ministers Ramsay MacDonald, Stanley Baldwin and most notably Neville Chamberlain towards Nazi Germany and Fascist Italy. Under British pressure, appeasement of Nazism and Fascism also played a role in French foreign policy of the period but was always much less popular there than in the United Kingdom.
Appendix may refer to:.
Apprenticeship is a system for training potential new practitioners of a trade or profession with on-the-job training and often some accompanying study. Apprenticeships may also enable practitioners to gain a license to practice in a regulated occupation. Most of their training is done while working for an experienced employer who helps the apprentices learn their trade or profession, in exchange for their continued labor for an agreed period after they have achieved measurable competencies.
An aquifer is an underground layer of water-bearing material consisting of permeable or fractured rock, or of unconsolidated materials. Aquifers vary greatly in their characteristics. The study of water flow in aquifers and the characterization of aquifers is called hydrogeology. Related concepts include aquitard, a bed of low permeability along an aquifer, and aquiclude, a solid and impermeable region underlying or overlying an aquifer, the pressure of which could lead to the formation of a confined aquifer. Aquifers can be classified as saturated versus unsaturated; aquifers versus aquitards; confined versus unconfined; isotropic versus anisotropic; porous, karst, or fractured; and transboundary aquifer.
Arable relates to the growing of crops:Arable farming or agronomy, the cultivation of field crops
Arable land, land upon which crops are cultivated
Arable crops program, a consolidated support system operated under the EU Common Agricultural Policy
Fivehead Arable Fields, a Site of Special Scientific Interest in Somerset, England.
An arbiter or arbitrator is a person by whose decision the parties to a dispute agree to be bound in arbitration.
Arboreal locomotion is the locomotion of animals in trees. In habitats in which trees are present, animals have evolved to move in them. Some animals may scale trees only occasionally (scansorial), but others are exclusively arboreal. The habitats pose numerous mechanical challenges to animals moving through them and lead to a variety of anatomical, behavioral and ecological consequences as well as variations throughout different species. Furthermore, many of these same principles may be applied to climbing without trees, such as on rock piles or mountains.
Arcade most often refers to:Arcade game, a coin-operated video, pinball, electro-mechanical, redemption, etc., game
Arcade video game, a coin-operated video game
Arcade cabinet, housing which holds an arcade video game's hardware
Arcade system board, a standardized printed circuit board
Amusement arcade, a place with arcade games.

The concept of an archetype appears in areas relating to behavior, historical psychology, philosophy and literary analysis.
The Arctic is the polar region of Earth that surrounds the North Pole, lying north of the Arctic Circle. The Arctic region, from the IERS Reference Meridian travelling east, consists of parts of northern Norway, northernmost Sweden, northern Finland, Russia, the United States (Alaska), Canada, Danish Realm (Greenland), and northern Iceland, along with the Arctic Ocean and adjacent seas.
Ardor or Ardour may refer to:Ardor (album), a 1994 album by Love Spirals Downwards
Ardour (album), a 2010 album by instrumental hip hop/electronica producer Teebs
Ardor (film), a 2002 South Korean film, also known as Milae
Ardour (software), a hard disk recorder and digital audio workstation application
Ardour (river), a river in southwestern France
Ardore, a town in Calabria, Italy
Ardor: The Book of the Dead Man, Vol. 2, a book of poems by Marvin Bell
Ada or Ardor: A Family Chronicle, a novel by Vladimir Nabokov.
A cant is the jargon or language of a group, often employed to exclude or mislead people outside the group. It may also be called a cryptolect, argot, pseudo-language, anti-language or secret language. Each term differs slightly in meaning; their uses are inconsistent.
Aridity is the condition of geographical regions which make up approximately 43% of total global available land area, characterized by low annual precipitation, increased temperatures, and limited water availability. These areas tend to fall upon degraded soils, and their health and functioning are key necessities of regulating ecosystems’ atmospheric components.
Aristocracy is a form of government that places power in the hands of a small, privileged ruling class, the aristocrats.
An armistice is a formal agreement of warring parties to stop fighting. It is not necessarily the end of a war, as it may constitute only a cessation of hostilities while an attempt is made to negotiate a lasting peace. It is derived from the Latin arma, meaning "arms" and -stitium, meaning "a stopping".
Aromatherapy is a practice based on the use of aromatic materials, including essential oils and other aroma compounds, with claims for improving psychological well-being. It is used as a complementary therapy or as a form of alternative medicine, and typically is used via inhalation and not by ingestion.
Arrhythmias, also known as cardiac arrhythmias, are irregularities in the heartbeat, including when it is too fast or too slow. Essentially, this is anything but normal sinus rhythm. A resting heart rate that is too fast – above 100 beats per minute in adults – is called tachycardia, and a resting heart rate that is too slow – below 60 beats per minute – is called bradycardia. Some types of arrhythmias have no symptoms. Symptoms, when present, may include palpitations or feeling a pause between heartbeats. In more serious cases, there may be lightheadedness, passing out, shortness of breath, chest pain, or decreased level of consciousness. While most cases of arrhythmia are not serious, some predispose a person to complications such as stroke or heart failure. Others may result in sudden death.
Arsenic is a chemical element; it has the symbol As and atomic number 33. It is a metalloid and one of the pnictogens, and therefore shares many properties with its group 15 neighbors phosphorus and antimony. Arsenic is notoriously toxic. It occurs naturally in many minerals, usually in combination with sulfur and metals, but also as a pure elemental crystal. It has various allotropes, but only the grey form, which has a metallic appearance, is important to industry.
Artifact or artefact may refer to:.
An artisan is a skilled craft worker who makes or creates material objects partly or entirely by hand. These objects may be functional or strictly decorative, for example furniture, decorative art, sculpture, clothing, food items, household items, and tools and mechanisms such as the handmade clockwork movement of a watchmaker. Artisans practice a craft and may through experience and aptitude reach the expressive levels of an artist.
Ascent or The Ascent may refer to:.
An ashram is a spiritual hermitage or a monastery in Hinduism.
Asphalt most often refers to:Bitumen, also known as "liquid asphalt cement" or simply "asphalt", a viscous form of petroleum mainly used as a binder in asphalt concrete
Asphalt concrete, a mixture of bitumen with coarse and fine aggregates, used as a road surface.
Aspiration or aspirations may refer to:.
An assay is an investigative (analytic) procedure in laboratory medicine, mining, pharmacology, environmental biology and molecular biology for qualitatively assessing or quantitatively measuring the presence, amount, or functional activity of a target entity. The measured entity is often called the analyte, the measurand, or the target of the assay. The analyte can be a drug, biochemical substance, chemical element or compound, or cell in an organism or organic sample. An assay usually aims to measure an analyte's intensive property and express it in the relevant measurement unit.
An astrolabe is an astronomical instrument dating to ancient times. It serves as a star chart and physical model of the visible half-dome of the sky. Its various functions also make it an elaborate inclinometer and an analog calculation device capable of working out several kinds of problems in astronomy. In its simplest form it is a metal disc with a pattern of wires, cutouts, and perforations that allows a user to calculate astronomical positions precisely. It is able to measure the altitude above the horizon of a celestial body, day or night; it can be used to identify stars or planets, to determine local latitude given local time, to survey, or to triangulate. It was used in classical antiquity, the Byzantine Empire, the Islamic Golden Age, the European Middle Ages and the Age of Discovery for all these purposes.
Asylum may refer to:.
An atoll is a ring-shaped island, including a coral rim that encircles a lagoon. There may be coral islands or cays on the rim. Atolls are located in warm tropical or subtropical parts of the oceans and seas where corals can develop. Most of the approximately 440 atolls in the world are in the Pacific Ocean.
In physics, attenuation is the gradual loss of flux intensity through a medium. For instance, dark glasses attenuate sunlight, lead attenuates X-rays, and water and air attenuate both light and sound at variable attenuation rates.
An auction is usually a process of buying and selling goods or services by offering them up for bids, taking bids, and then selling the item to the highest bidder or buying the item from the lowest bidder. Some exceptions to this definition exist and are described in the section about different types. The branch of economic theory dealing with auction types and participants' behavior in auctions is called auction theory.
Auditory means of or relating to the process of hearing:Auditory system, the neurological structures and pathways of sound perception
Auditory bulla, part of auditory system found in mammals other than primates
Auditory nerve, also known as the cochlear nerve is one of two parts of a cranial nerve
Auditory ossicles, three bones in the middle ear that transmit sounds
Hearing (sense), the auditory sense, the sense by which sound is perceived
Ear, the auditory end organ
Cochlea, the auditory branch of the inner ear
Sound, the physical signal perceived by the auditory system
External auditory meatus, the ear canal
Primary auditory cortex, the part of the higher-level of the brain that serves hearing
Auditory agnosia
Auditory exclusion, a form of temporary hearing loss under high stress
Auditory feedback, an aid to control speech production and singing
Auditory hallucination, perceiving sounds without auditory stimulus
Auditory illusion, sound trick analogous to an optical illusion
Auditory imagery, hearing in head in the absence of sound
Auditory learning, learning by listening
Auditory phonetics, the science of the sounds of language
Auditory scene analysis, the process by which a scene containing many sounds is perceived
Auditory science, concerning the perception of sound

.
Augury was a Greco-Roman religious practice of observing the behavior of birds, to receive omens. When the individual, known as the augur, read these signs, it was referred to as "taking the auspices". "Auspices" means "looking at birds". Auspex, another word for augur, can be translated to "one who looks at birds". Depending upon the birds, the auspices from the gods could be favorable or unfavorable. Sometimes politically motivated augurs would fabricate unfavorable auspices in order to delay certain state functions, such as elections. Pliny the Elder attributes the invention of auspicy to Tiresias the seer of Thebes.
An aureola or aureole is the radiance of luminous cloud which, in paintings of sacred personages, surrounds the whole figure.
Autarky is the characteristic of self-sufficiency, usually applied to societies, communities, states, and their economic systems.
An autoclave is a machine used to carry out industrial and scientific processes requiring elevated temperature and pressure in relation to ambient pressure and temperature. Autoclaves are found in many medical settings, laboratories, and other places that need to ensure the sterility of an object.
An autopsy is a surgical procedure that consists of a thorough examination of a corpse by dissection to determine the cause, mode, and manner of death; or the exam may be performed to evaluate any disease or injury that may be present for research or educational purposes. The term necropsy is generally used for non-human animals.
Auxins are a class of plant hormones with some morphogen-like characteristics. Auxins play a cardinal role in coordination of many growth and behavioral processes in plant life cycles and are essential for plant body development. The Dutch biologist Frits Warmolt Went first described auxins and their role in plant growth in the 1920s.
Kenneth V. Thimann became the first to isolate one of these phytohormones and to determine its chemical structure as indole-3-acetic acid (IAA). Went and Thimann co-authored a book on plant hormones, Phytohormones, in 1937.
Greed is an insatiable desire for material gain or social value, such as status or power.
Aviation includes the activities surrounding mechanical flight and the aircraft industry. Aircraft include fixed-wing and rotary-wing types, morphable wings, wing-less lifting bodies, as well as lighter-than-air aircraft such as hot air balloons and airships.
Avulsion in general refers to a tearing away. Specifically, it can refer to:Avulsion fracture, when a fragment of bone tears away from the main mass of bone as a result of physical trauma
Avulsion injury, in which a body structure is detached from its normal point of insertion, either torn away by trauma or cut by surgery
Avulsion, the sudden loss of land by the action of water
Avulsion (river), abandonment of an old river channel and the formation of a new one

.
Axial may refer to:one of the anatomical directions describing relationships in an animal body
In geometry:a geometric term of location
an axis of rotationIn chemistry, referring to an axial bond
a type of modal frame, in music
axial-flow, a type of fan
the Axial Age in China, India, etc.
Axial Seamount and submarine volcano off Oregon, USA
Axial, Colorado, a ghost town.
An azimuth is the horizontal angle from a cardinal direction, most commonly north, in a local or observer-centric spherical coordinate system.
A bactericide or bacteriocide, sometimes abbreviated Bcidal, is a substance which kills bacteria. Bactericides are disinfectants, antiseptics, or antibiotics.
However, material surfaces can also have bactericidal properties based solely on their physical surface structure, as for example biomaterials like insect wings.
A bailiff is a manager, overseer or custodian – a legal officer to whom some degree of authority or jurisdiction is given. There are different kinds, and their offices and scope of duties vary.
Bakelite, formally poly­oxy­benzyl­methylen­glycol­anhydride, is a thermosetting phenol formaldehyde resin, formed from a condensation reaction of phenol with formaldehyde. The first plastic made from synthetic components, it was developed by Belgian chemist Leo Baekeland in Yonkers, New York, in 1907, and patented on December 7, 1909.
A balcony is a platform projecting from the wall of a building, supported by columns or console brackets, and enclosed with a balustrade, usually above the ground floor. They are commonly found on multi-level houses, apartments, and cruise ships.
Ballast is dense material used as a weight to provide stability to a vehicle or structure. Ballast, other than cargo, may be placed in a vehicle, often a ship or the gondola of a balloon or airship, to provide stability. A compartment within a boat, ship, submarine, or other floating structure that holds water is called a ballast tank. Water can be moved in and out from the ballast tank to balance the ship. In a vessel that travels on the water, the ballast will be kept below the water level, to counteract the effects of weight above the water level. The ballast may be redistributed in the vessel or disposed of altogether to change its effects on the movement of the vessel.
A banyan, also spelled banian, is a fig that develops accessory trunks from adjacent prop roots, allowing the tree to spread outwards indefinitely. This distinguishes banyans from other trees with a strangler habit that begin life as an epiphyte, i.e. a plant that grows on another plant, when its seed germinates in a crack or crevice of a host tree or edifice. "Banyan" often specifically denotes Ficus benghalensis, which is the national tree of India, though the name has also been generalized to denominate all figs that share a common life cycle and used systematically in taxonomy to denominate the subgenus Urostigma.
Barbarism, barbarity, or barbarous may refer to:Barbarism (linguistics), a non-standard word, expression, or pronunciation
Hybrid words, formerly called "barbarisms"
Any society construed as barbarian
Barbarian invasions, a period of migrations within or into Europe in the middle of the first millennium AD.
A baritone is a type of classical male singing voice whose vocal range lies between the bass and the tenor voice-types. It is the most common male voice. The term originates from the Greek βαρύτονος (barýtonos), meaning "low sounding". Composers typically write music for this voice in the range from the second F below middle C to the F above middle C (i.e. F2–F4) in choral music, and from the second G below middle C to the G above middle C (G2 to G4) in operatic music, but the range can extend at either end. Subtypes of baritone include the baryton-Martin baritone (light baritone), lyric baritone, Kavalierbariton, Verdi baritone, dramatic baritone, baryton-noble baritone, and the bass-baritone.
Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally. One of the first cultivated grains, it was domesticated in the Fertile Crescent around 9000 BC, giving it nonshattering spikelets and making it much easier to harvest. Its use then spread throughout Eurasia by 2000 BC. Barley prefers relatively low temperatures and well-drained soil to grow. It is relatively tolerant of drought and soil salinity, but is less winter-hardy than wheat or rye.
The Baroque is a Western style of architecture, music, dance, painting, sculpture, poetry, and other arts that flourished from the early 17th century until the 1750s. It followed Renaissance art and Mannerism and preceded the Rococo and Neoclassical styles. It was encouraged by the Catholic Church as a means to counter the simplicity and austerity of Protestant architecture, art, and music, though Lutheran Baroque art developed in parts of Europe as well.
A barrister is a type of lawyer in common law jurisdictions that originated from the Inns of Court in the medieval English legal system. Barristers mostly specialise in courtroom advocacy and litigation. Their tasks include arguing cases in courts and tribunals, drafting legal pleadings, researching the law and giving legal opinions.
A basement is any floor of a building that is not above the grade plane. Especially in residential buildings, it often is used as a utility space for a building, where such items as the furnace, water heater, breaker panel or fuse box, car park, and air-conditioning system are located; so also are amenities such as the electrical system and cable television distribution point. In cities with high property prices, such as London, basements are often fitted out to a high standard and used as living space.
In Ancient Roman architecture, a basilica was a large public building with multiple functions that was typically built alongside the town's forum. The basilica was in the Latin West equivalent to a stoa in the Greek East. The building gave its name to the basilica architectural form.
A bastion is a structure projecting outward from the curtain wall of a fortification, most commonly angular in shape and positioned at the corners of the fort. The fully developed bastion consists of two faces and two flanks, with fire from the flanks being able to protect the curtain wall and the adjacent bastions. Compared with the medieval fortified towers they replaced, bastion fortifications offered a greater degree of passive resistance and more scope for ranged defence in the age of gunpowder artillery. As military architecture, the bastion is one element in the style of fortification dominant from the mid 16th to mid 19th centuries.
Bauxite is a sedimentary rock with a relatively high aluminium content. It is the world's main source of aluminium and gallium. Bauxite consists mostly of the aluminium minerals gibbsite, boehmite, and diaspore, mixed with the two iron oxides goethite and hematite, the aluminium clay mineral kaolinite and small amounts of anatase and ilmenite .
Bauxite appears dull in luster and is reddish-brown, white, or tan.
A beacon is an intentionally conspicuous device designed to attract attention to a specific location. A common example is the lighthouse, which draws attention to a fixed point that can be used to navigate around obstacles or into port. More modern examples include a variety of radio beacons that can be read on radio direction finders in all weather, and radar transponders that appear on radar displays.
Beetles are insects that form the order Coleoptera, in the superorder Holometabola. Their front pair of wings are hardened into wing-cases, elytra, distinguishing them from most other insects. The Coleoptera, with about 400,000 described species, is the largest of all orders, constituting almost 40% of described arthropods and 25% of all known animal species; new species are discovered frequently, with estimates suggesting that there are between 0.9 and 2.1 million total species.
The belfry is a structure enclosing bells for ringing as part of a building, usually as part of a bell tower or steeple. It can also refer to the entire tower or building, particularly in continental Europe for such a tower attached to a city hall or other civic building.
A belief is a subjective attitude that something is true or a state of affairs is the case. A subjective attitude is a mental state of having some stance, take, or opinion about something. In epistemology, philosophers use the term belief to refer to attitudes about the world which can be either true or false. To believe something is to take it to be true; for instance, to believe that snow is white is comparable to accepting the truth of the proposition "snow is white". However, holding a belief does not require active introspection. For example, few individuals carefully consider whether or not the sun will rise the next morning, simply assuming that it will. Moreover, beliefs need not be occurrent, but can instead be dispositional.
The benthic zone is the ecological region at the lowest level of a body of water such as an ocean, lake, or stream, including the sediment surface and some sub-surface layers. The name comes from the Ancient Greek word βένθος (bénthos), meaning "the depths". Organisms living in this zone are called benthos and include microorganisms as well as larger invertebrates, such as crustaceans and polychaetes.
Beryl ( BERR-əl) is a mineral composed of beryllium aluminium silicate with the chemical formula Be3Al2(SiO3)6. Well-known varieties of beryl include emerald and aquamarine. Naturally occurring hexagonal crystals of beryl can be up to several meters in size, but terminated crystals are relatively rare. Pure beryl is colorless, but it is frequently tinted by impurities; possible colors are green, blue, yellow, pink, and red (the rarest). It is an ore source of beryllium.
Bifurcation or bifurcated may refer to:.
In European militaries, a billet is a living-quarters to which a soldier is assigned to sleep. In American usage, it refers to a specific personnel position, assignment, or duty station to which a soldier can be assigned. Historically, a billet was a private dwelling that was required to accept a soldier.
Biomechanics is the study of the structure, function and motion of the mechanical aspects of biological systems, at any level from whole organisms to organs, cells and cell organelles, and even proteins using the methods of mechanics. Biomechanics is a branch of biophysics.
A biopsy is a medical test commonly performed by a surgeon, an interventional radiologist, or an interventional cardiologist. The process involves the extraction of sample cells or tissues for examination to determine the presence or extent of a disease. The tissue is then fixed, dehydrated, embedded, sectioned, stained and mounted before it is generally examined under a microscope by a pathologist; it may also be analyzed chemically. When an entire lump or suspicious area is removed, the procedure is called an excisional biopsy. An incisional biopsy or core biopsy samples a portion of the abnormal tissue without attempting to remove the entire lesion or tumor. When a sample of tissue or fluid is removed with a needle in such a way that cells are removed without preserving the histological architecture of the tissue cells, the procedure is called a needle aspiration biopsy. Biopsies are most commonly performed for insight into possible cancerous or inflammatory conditions.
Birth rate, also known as natality and crude birth rate, is the total number of live human births per 1,000 population for a given period divided by the length of the period in years. The number of live births is normally taken from a universal registration system for births, or population counts from a census. The birth rate is used to calculate population growth. The estimated average population may be taken as the mid-year population.
In church governance, a diocese or bishopric is the ecclesiastical district under the jurisdiction of a bishop.
Bitumen is an immensely viscous constituent of petroleum. Depending on its exact composition, it can be a sticky, black liquid or an apparently solid mass that behaves as a liquid over very large time scales. In American English, the material is commonly referred to as asphalt. Whether found in natural deposits or refined from petroleum, the substance is classed as a pitch. Prior to the 20th century, the term asphaltum was in general use. The word derives from the Ancient Greek word ἄσφαλτος (ásphaltos), which referred to natural bitumen or pitch. The largest natural deposit of bitumen in the world is the Pitch Lake of southwest Trinidad, which is estimated to contain 10 million tons.
Blight is a specific symptom affecting plants in response to infection by a pathogenic organism.
A blizzard is a severe snowstorm characterized by strong sustained winds and low visibility, lasting for a prolonged period of time—typically at least three or four hours. A ground blizzard is a weather condition where snow that has already fallen is being blown by wind. Blizzards can have an immense size and usually stretch to hundreds or thousands of kilometres.
A blockade is the act of actively preventing a country or region from receiving or sending out food, supplies, weapons, or communications, and sometimes people, by military force.
A blockade differs from an embargo or sanction, which are legal barriers to trade rather than physical barriers. It is also distinct from a siege in that a blockade is usually directed at an entire country or region, rather than a fortress or city and the objective may not always be to conquer the area.
Bloom or blooming may refer to:.
In Buddhism, a bodhisattva is a person who has attained, or is striving towards, bodhi or Buddhahood. Often, the term specifically refers to a person who forgoes or delays personal nirvana or bodhi in order to compassionately help other individuals reach Buddhahood.
Bole may refer to:.
Bondage may refer to:.
A borough is an administrative division in various English-speaking countries. In principle, the term borough designates a self-governing walled town, although in practice, official use of the term varies widely.
A botnet is a group of Internet-connected devices, each of which runs one or more bots. Botnets can be used to perform distributed denial-of-service (DDoS) attacks, steal data, send spam, and allow the attacker to access the device and its connection. The owner can control the botnet using command and control (C&C) software. The word "botnet" is a portmanteau of the words "robot" and "network". The term is usually used with a negative or malicious connotation.
In geology, a boulder is a rock fragment with size greater than 25.9 cm (10.2 in) in diameter. Smaller pieces are called cobbles and pebbles. While a boulder may be small enough to move or roll manually, others are extremely massive. In common usage, a boulder is too large for a person to move. Smaller boulders are usually just called rocks or stones.
Bounty or bounties commonly refers to:Bounty (reward), an amount of money or other reward offered by an organization for a specific task done with a person or thing.
Bovidae is the biological family of cloven-hoofed, ruminant mammals that includes cattle, bison, buffalo, antelopes, and goat-antelopes such as sheep and goats. There are 143 extant species and 300 known extinct species of bovids, which are divided into either 11 major subfamilies, or two subfamilies with thirteen tribes. The earliest known bovid had evolved by 20 million years ago, in the early Miocene.
In botany, a bract is a modified or specialized leaf, associated with a reproductive structure such as a flower, inflorescence axis or cone scale.
Bracts are usually different from foliage leaves in size, color, shape or texture. They also look different from the parts of the flower, such as the petals or sepals.
Brass is an alloy of copper and zinc, in proportions which can be varied to achieve different colours and mechanical, electrical, acoustic, and chemical properties, but copper typically has the larger proportion, generally 2⁄3 copper and 1⁄3 zinc. In use since prehistoric times, it is a substitutional alloy; atoms of the two constituents may replace each other within the same crystal structure.
Breach, Breached, or The Breach may refer to:.
Brevity is concision or brevitas, the quality of being brief or concise, or:Brevity , a comic strip created by Guy Endore-Kaiser and Rodd Perry
Brevity code, a vocal word replacement system
Operation Brevity, a World War II battle

.
A brigade is a major tactical military formation that typically comprises three to six battalions plus supporting elements. It is roughly equivalent to an enlarged or reinforced regiment. Two or more brigades may constitute a division.
Brine is a high-concentration solution of salt in water. In diverse contexts, brine may refer to the salt solutions ranging from about 3.5% up to about 26%. Brine forms naturally due to evaporation of ground saline water but it is also generated in the mining of sodium chloride. Brine is used for food processing and cooking, for de-icing of roads and other structures, and in a number of technological processes. It is also a by-product of many industrial processes, such as desalination, so it requires wastewater treatment for proper disposal or further utilization.
A bristle is a stiff hair or feather, either on an animal, such as a pig, a plant, or on a tool such as a brush or broom.
A bronchus is a passage or airway in the lower respiratory tract that conducts air into the lungs. The first or primary bronchi to branch from the trachea at the carina are the right main bronchus and the left main bronchus. These are the widest bronchi, and enter the right lung, and the left lung at each hilum. The main bronchi branch into narrower secondary bronchi or lobar bronchi, and these branch into narrower tertiary bronchi or segmental bronchi. Further divisions of the segmental bronchi are known as 4th order, 5th order, and 6th order segmental bronchi, or grouped together as subsegmental bronchi.
The bronchi, when too narrow to be supported by cartilage, are known as bronchioles. No gas exchange takes place in the bronchi.
Bronze is an alloy consisting primarily of copper, commonly with about 12–12.5% tin and often with the addition of other metals and sometimes non-metals or metalloids. These additions produce a range of alloys some of which are harder than copper alone or have other useful properties, such as strength, ductility, or machinability.
Brotherhood or The Brotherhood may refer to:.
Bureau may refer to:.
Burgundy is a historical region in France, encompassing the territory of the former administrative region of the same name, that existed from 1982 to 2015, and was merged since 1 January 2016 into the newly created administrative region of Bourgogne-Franche-Comté, encompassing its western half. In historical terms, that region was formed as the Duchy of Burgundy, which existed between the 10th and the 18th century. During the late medieval and early modern periods, the region was of great political importance, being the core of the Valois-Burgundian State, and also becoming a focal point of diplomacy and courtly culture that set the fashion for European royal houses and their courts. The regional capital, Dijon, was wealthy and powerful, being a major European centre of art and science, and of Western Monasticism.
Burial, also known as interment or inhumation, is a method of final disposition whereby a dead body is placed into the ground, sometimes with objects. This is usually accomplished by excavating a pit or trench, placing the deceased and objects in it, and covering it over. A funeral is a ceremony that accompanies the final disposition.
Hessian, burlap in North America, or crocus in The Caribbean, is a woven fabric made of vegetable fibres: usually the skin of the jute plant, or sisal leaves. It is generally used for rough handling, such as to make sacks in which to ship farm products and sandbags, and for wrapping tree-root balls. However, this dense woven fabric, historically coarse, is also recently being produced in a more refined state—where it is known simply as jute—so as to provide an eco-friendly material for bags, rugs, and other products.
A bushel is an imperial and US customary unit of volume, based upon an earlier measure of dry capacity. The old bushel was used mostly for agricultural products, such as wheat: in modern usage, the volume is nominal, with bushels denoting a mass defined differently for each commodity.
In geomorphology, a butte is an isolated hill with steep, often vertical sides and a small, relatively flat top; buttes are smaller landforms than mesas, plateaus, and tablelands. The word butte comes from the French word butte, meaning 'knoll' ; its use is prevalent in the Western United States, including the Southwest, where mesa is used for the larger landform.
A by-law, is a set of rules or law established by an organization or community so as to regulate itself, as allowed or provided for by some higher authority. The higher authority, generally a legislature or some other government body, establishes the degree of control that the by-laws may exercise. By-laws may be established by entities such as a business corporation, a neighbourhood association, or depending on the jurisdiction, a municipality.
A cabal is a group of people who are united in some close design, usually to promote their private views or interests in an ideology, a state, or another community, often by intrigue and usually without the knowledge of those who are outside their group. The use of this term usually carries negative connotations of political purpose, conspiracy and secrecy. It can also refer to a secret plot or a clique, or it may be used as a verb.
Cache, caching, or caché may refer to:.


.
A reef is a ridge or shoal of rock, coral, or similar relatively stable material lying beneath the surface of a natural body of water. Many reefs result from natural, abiotic (non-living) processes such as deposition of sand or wave erosion planing down rock outcrops. However, reefs such as the coral reefs of tropical waters are formed by biotic (living) processes, dominated by corals and coralline algae. Artificial reefs, such as shipwrecks and other man-made underwater structures, may occur intentionally or as the result of an accident. These are sometimes designed to increase the physical complexity of featureless sand bottoms to attract a more diverse range of organisms. They provide shelter to various aquatic animals which help prevent extinction. Another reason reefs are put in place is for aquaculture, and fish farmers who are looking to improve their businesses sometimes invest in them. Reefs are often quite near to the surface, but not all definitions require this.
In materials science, a refractory is a material that is resistant to decomposition by heat or chemical attack and that retains its strength and rigidity at high temperatures. They are inorganic, non-metallic compounds that may be porous or non-porous, and their crystallinity varies widely: they may be crystalline, polycrystalline, amorphous, or composite. They are typically composed of oxides, carbides or nitrides of the following elements: silicon, aluminium, magnesium, calcium, boron, chromium and zirconium. Many refractories are ceramics, but some such as graphite are not, and some ceramics such as clay pottery are not considered refractory. Refractories are distinguished from the refractory metals, which are elemental metals and their alloys that have high melting temperatures.
Boat racing (Regatta) is a sport in which boats, or other types of watercraft, race on water. Boat racing powered by oars is recorded as having occurred in ancient Egypt, and it is likely that people have engaged in races involving boats and other water-borne craft for as long as such watercraft have existed.
In a monarchy, a regent is a person appointed to execute the office of a monarch temporarily. Regencies may arise for a number of reasons, including the monarch being a minor, ill, absent from the country, or otherwise unavailable. A regent may also be appointed in cases where the throne is vacant, or the identity of the legitimate monarch is disputed.
In politics, a regime is a system of government that determines access to public office, and the extent of power held by officials. The two broad categories of regimes are democratic and autocratic. A key similarity across all regimes is the presence of rulers of both formal and informal institutions, which interact dynamically to adapt to changes to their environment.
The reindeer or caribou is a species of deer with circumpolar distribution, native to Arctic, subarctic, tundra, boreal, and mountainous regions of Northern Europe, Siberia, and North America. It is the only representative of the genus Rangifer. More recent studies suggest the splitting of reindeer and caribou. "All caribou and reindeer throughout the world are considered to be the same species, but there are 7 subspecies.".
A reliquary is a container for relics. A portable reliquary, or the room in which one is stored, may also be called a feretory. A brooch-like container for a very small relic may be called a "theca".
Remnant or remnants may refer to:.
The Renaissance is a European period of history and cultural movement, very roughly defined as covering the 14th through 17th centuries, though sometimes more narrowly defined for instance as only covering the 15th through 16th centuries. It marked the transition from the Middle Ages to modernity and was characterized by the European rediscovery and revival of the literary, philosophical, and artistic achievements of classical antiquity. Associated with great social change in most fields and disciplines, including art, architecture, politics, literature, exploration and science, the Renaissance was first centered in the Republic of Florence, then spread to the rest of Italy and later throughout Europe. The term rinascita ('rebirth') first appeared in Lives of the Artists by Giorgio Vasari, while the corresponding French word renaissance was adopted into English as the term for this period during the 1830s.
Reptiles, as commonly defined, are tetrapod vertebrate animals with an ectothermic metabolism and amniotic development. Reptiles traditionally comprise four orders: Testudines (turtles), Crocodilia, Squamata and Rhynchocephalia (tuatara), with about 12,000 extant species listed in the Reptile Database. The study of the traditional reptile orders, customarily in combination with the study of modern amphibians, is called herpetology.
A reservoir is an enlarged lake behind a dam, usually built to store fresh water, often doubling for hydroelectric power generation.
Residue may refer to:.
A resin is a solid or highly viscous liquid that can be converted into a polymer. Resins may be biological or synthetic in origin, but are typically harvested from plants. Resins are mixtures of organic compounds insoluble in water, predominantly terpenes. Technically, resins should not be confused with gums, which consist predominantly of water-soluble polysaccharides, although these two terms are often interchangeable in the less formal context. Common resins include pine oleoresins, amber, hashish, frankincense, myrrh and the animal-derived resin, shellac. Resins are used in varnishes, adhesives, food additives, incenses and perfumes.
Restoration is the act of restoring something to its original state. This may refer to:Conservation and restoration of cultural property
Audio restoration
Conservation and restoration of immovable cultural property
Film restoration
Image restoration
Textile restoration
Ecological restoration.
A reticle or reticule, also known as a graticule or crosshair, is a pattern of fine lines or markings built into the eyepiece of an optical device such as a telescopic sight, spotting scope, theodolite, optical microscope or the screen of an oscilloscope, to provide measurement references during visual inspections. Today, engraved lines or embedded fibers may be replaced by a digital image superimposed on a screen or eyepiece. Both terms may be used to describe any set of patterns used for aiding visual measurements and calibrations, but in modern use reticle is most commonly used for weapon sights, while graticule is more widely used for non-weapon measuring instruments such as oscilloscope display, astronomic telescopes, microscopes and slides, surveying instruments and other similar devices.
A retinue is a body of persons "retained" in the service of a noble, royal personage, or dignitary; a suite of retainers.
Revelry may refer to:The revelries of Saturnalia
The Revelry (album), by the Bullets and Octane
Revelry
"Revelry" (song), by Kings of Leon
"Revelry", a song by Yachts (band)
Party

.
Reverie may refer to:A daydream or a dreamy state.
W. R. Bion's psychoanalytic use of "reverie".
Rhapsody may refer to:.
Rhenium is a chemical element; it has symbol Re and atomic number 75. It is a silvery-gray, heavy, third-row transition metal in group 7 of the periodic table. With an estimated average concentration of 1 part per billion (ppb), rhenium is one of the rarest elements in the Earth's crust. It has one of the highest melting and boiling points of any element. It resembles manganese and technetium chemically and is mainly obtained as a by-product of the extraction and refinement of molybdenum and copper ores. It shows in its compounds a wide variety of oxidation states ranging from −3 to +7.
Rhododendron, from Ancient Greek ῥόδον (rhódon), meaning "rose", and δένδρον (déndron), meaning "tree", is a very large genus of about 1,024 species of woody plants in the heath family (Ericaceae). They can be either evergreen or deciduous. Most species are native to eastern Asia and the Himalayan region, but smaller numbers occur elsewhere in Asia, and in North America, Europe and Australia.
Rhodonite is a manganese inosilicate, with the formula (Mn, Fe, Mg, Ca)SiO3, and member of the pyroxenoid group of minerals, crystallizing in the triclinic system. The term rhodonite was first introduced by Germar. from Ancient Greek  ῥόδον (rhódon) 'rose'. It commonly occurs as cleavable to compact masses with a rose-red color often tending to brown due to surface oxidation. The rose-red hue is caused by the manganese cation (Mn2+).
Rhubarb is the fleshy, edible stalks (petioles) of species and hybrids of Rheum in the family Polygonaceae, which are cooked and used for food. The plant is a herbaceous perennial that grows from short, thick rhizomes. Historically, different plants have been called "rhubarb" in English. The large, triangular leaves contain high levels of oxalic acid and anthrone glycosides, making them poisonous and therefore inedible. The small flowers are grouped in large compound leafy greenish-white to rose-red inflorescences.
The Rialto is a central area of Venice, Italy, in the sestiere of San Polo. It is, and has been for many centuries, the financial and commercial heart of the city. Rialto is known for its prominent markets as well as for the monumental Rialto Bridge across the Grand Canal.
A ribosome is a ribonucleoprotein particle found in all cells, both prokaryotic and eukaryotic, responsible for the synthesis of proteins. A ribosome functions as a molecular machine in the translation of strands of messenger RNA (mRNA) and production of a protein. A ribosome links amino acids together in the order specified by the codons of mRNA molecules to form polypeptide chains. A ribosome is made up of a large and a small subunit, each consisting of one or more ribosomal RNA molecules and many ribosomal proteins. The ribosomes and associated molecules are also known as the translational apparatus.
A ridge is a long, narrow, elevated geomorphologic landform, structural feature, or a combination of both separated from the surrounding terrain by steep sides. The sides of a ridge slope away from a narrow top, the crest or ridgecrest, with the terrain dropping down on either side. The crest, if narrow, is also called a ridgeline. Limitations on the dimensions of a ridge are lacking. Its height above the surrounding terrain can vary from less than a meter to hundreds of meters. A ridge can be either depositional, erosional, tectonic, or a combination of these in origin and can consist of either bedrock, loose sediment, lava, or ice depending on its origin. A ridge can occur as either an isolated, independent feature or part of a larger geomorphological and/or structural feature. Frequently, a ridge can be further subdivided into smaller geomorphic or structural elements.
In geology, a rift is a linear zone where the lithosphere is being pulled apart and is an example of extensional tectonics. Typical rift features are a central linear downfaulted depression, called a graben, or more commonly a half-graben with normal faulting and rift-flank uplifts mainly on one side. Where rifts remain above sea level they form a rift valley, which may be filled by water forming a rift lake. The axis of the rift area may contain volcanic rocks, and active volcanism is a part of many, but not all, active rift systems.
Rigging comprises the system of ropes, cables and chains, which support and control a sailing ship or sail boat's masts and sails. Standing rigging is the fixed rigging that supports masts including shrouds and stays. Running rigging is rigging which adjusts the position of the vessel's sails and spars including halyards, braces, sheets and vangs.
In hillslope geomorphology, a rill is a shallow channel cut into soil by the erosive action of flowing surface water. Similar but smaller incised channels are known as microrills; larger incised channels are known as gullies.
Rime may refer to:Rime ice, ice that forms when water droplets in fog freeze to the outer surfaces of objects, such as trees.
A stream is a continuous body of surface water flowing within the bed and banks of a channel. Depending on its location or certain characteristics, a stream may be referred to by a variety of local or regional names. Long, large streams are usually called rivers, while smaller, less voluminous and more intermittent streams are known, amongst others, as brook, creek, rivulet, rill, run, tributary, feeder, freshet, narrow river, and streamlet.
A road is a thoroughfare from one place to another, primarily used for movement of traffic. Many roads are paved.

.
An easel is an upright support used for displaying and/or fixing something resting upon it, at an angle of about 20° to the vertical. In particular, painters traditionally use an easel to support a painting while they work on it, normally standing up; easels are also sometimes used to display finished paintings and prints. Artists' easels are still typically made of wood, in functional designs that have changed little for centuries, or even millennia,
though new materials and designs exist. Easels are typically made from wood, aluminum or steel.
Echelon may refer to:.
Could not find summary for "Eddyline".
Effluent is wastewater from sewers or industrial outfalls that flows directly into surface waters, either untreated or after being treated at a facility. The term has slightly different meanings in certain contexts, and may contain various pollutants depending on the source.
Egress may refer to:Ingress, egress, and regress, legal terms referring to an individual's right to travel or move
Egress, the passage of electromagnetic fields through the shield of a coaxial cable
Egress filtering, in computer networking, monitoring and/or restricting the flow of outbound information
Egress Software, a British provider of data security services
Interior and exterior egress, two of the four contacts observed during an astronomical transit.
Sambucus is a genus of between 20 and 30 species of flowering plants in the family Adoxaceae. The various species are commonly referred to as elder, with the flowers as elderflower, and the fruit as elderberry.
Embankment may refer to:.
Emissary may refer to:.
An emulsion is a mixture of two or more liquids that are normally immiscible owing to liquid-liquid phase separation. Emulsions are part of a more general class of two-phase systems of matter called colloids. Although the terms colloid and emulsion are sometimes used interchangeably, emulsion more narrowly refers to when both phases, dispersed and continuous, are liquids. In an emulsion, one liquid is dispersed in the other. Examples of emulsions include vinaigrettes, homogenized milk, liquid biomolecular condensates, and some cutting fluids for metal working.
Enamel may refer to:.
An enclave is a territory that is entirely surrounded by the territory of only one other state or entity. An enclave can be an independent territory or part of a larger one. Enclaves may also exist within territorial waters. Enclave is sometimes used improperly to denote a territory that is only partly surrounded by another state. Enclaves that are not part of a larger territory are not exclaves, for example Lesotho, San Marino and Vatican City are enclaved sovereign states.
The endosperm is a tissue produced inside the seeds of most of the flowering plants following double fertilization. It is triploid in most species, which may be auxin-driven. It surrounds the embryo and provides nutrition in the form of starch, though it can also contain oils and protein. This can make endosperm a source of nutrition in animal diet. For example, wheat endosperm is ground into flour for bread, while barley endosperm is the main source of sugars for beer production. Other examples of endosperm that forms the bulk of the edible portion are coconut "meat" and coconut "water", and corn. Some plants, such as certain orchids, lack endosperm in their seeds.
Engram may refer to:Engram (neuropsychology), a physical means by which memory traces are stored
Engram (Dianetics), a term used in Scientology and Dianetics for a "recording" of a past painful event not normally accessible to the conscious mind
Engram (album), a 2009 album by black metal band Beherit
Engram (film), a 2014 short film.
An entourage is an informal group or band of people who are closely associated with a (usually) famous, notorious, or otherwise notable individual. The word can also refer to:.
Epaulette is a type of ornamental shoulder piece or decoration used as insignia of rank by armed forces and other organizations. Flexible metal epaulettes are referred to as shoulder scales.
An epitaph is a short text honoring a deceased person. Strictly speaking, it refers to text that is inscribed on a tombstone or plaque, but it may also be used in a figurative sense. Some epitaphs are specified by the person themselves before their death, while others are chosen by those responsible for the burial. An epitaph may be written in prose or in verse.

An equerry is an officer of honour. Historically, it was a senior attendant with responsibilities for the horses of a person of rank. In contemporary use, it is a personal attendant, usually upon a sovereign, a member of a royal family, or a national representative. The role is equivalent to an aide-de-camp, but the term is prevalent only among some members of the Commonwealth of Nations.
Ermine may refer to three species of mustelid in the genus Mustela, or their fur:Stoat or Eurasian ermine, Mustela erminea, found throughout Eurasia and northern North America
American ermine, Mustela richardsonii, found throughout North America aside from most of Alaska and the Arctic
Haida ermine, Mustela haidarum, endemic to Haida Gwaii and the Alexander Archipelago on the Pacific Northwest coast of North America.
An escarpment is a steep slope or long cliff that forms as a result of faulting or erosion and separates two relatively level areas having different elevations.
An esplanade or promenade is a long, open, level area, usually next to a river or large body of water, where people may walk. The historical definition of esplanade was a large, open, level area outside fortress or city walls to provide clear fields of fire for the fortress's guns. In modern usage, the space allows the area to be paved as a pedestrian walk; esplanades are often on sea fronts and allow walking whatever the state of the tide, without having to walk on the beach.

Abolition refers to the act of putting an end to something by law, and may refer to:Abolitionism, abolition of slavery
Abolition of the death penalty, also called capital punishment
Abolition of monarchy
Abolition of nuclear weapons
Abolition of prisons
Abolition of ICE
Police abolition movement
Abolition of suffering
Abolitionism, related to veganism
Abolition of time zones
Abolition of borders.
Abstract may refer to:"Abstract", a 2017 episode of the animated television series Adventure Time
Abstract (album), 1962 album by Joe Harriott
Abstract algebra, sets with specific operations acting on their elements
Abstract of title, a summary of the documents affecting the title to a parcel of land
Abstract (law), a summary of a legal document
Abstract (summary), in academic publishing
Abstract art, artistic works that do not attempt to represent reality or concrete subjects
Abstract: The Art of Design, 2017 Netflix documentary series
Abstract music, music that is non-representational
Abstract object in philosophy
Abstract structure in mathematics
Abstract type in computer science
The property of an abstraction
Q-Tip (musician), also known as "The Abstract"
Abstract and concrete
Hydrogen atom abstraction, in chemistry.
Absolution is a theological term for the forgiveness imparted by ordained Christian priests and experienced by Christian penitents. It is a universal feature of the historic churches of Christendom, although the theology and the practice of absolution vary between Christian denominations.
Accord may refer to:.
Acrimony may refer to:a feeling of hatred
Acrimony (band), a rock band
Acrimony (film), a 2018 film.
An adversary is generally considered to be a person, group, or force that opposes and/or attacks.
An advocate is a professional in the field of law. Different countries and legal systems use the term with somewhat differing meanings. The broad equivalent in many English law–based jurisdictions could be a barrister or a solicitor. However, in Scottish, Manx, South African, Italian, French, Spanish, Portuguese, Scandinavian, Polish, Israeli, South Asian and South American jurisdictions, "advocate" indicates a lawyer of superior classification.
Aesthetics is the branch of philosophy that studies beauty, taste, and related phenomena. In a broad sense, it includes the philosophy of art, which examines the nature of art, artistic creativity, the meanings of artworks, and audience appreciation.
Wealth is the abundance of valuable financial assets or physical possessions which can be converted into a form that can be used for transactions. This includes the core meaning as held in the originating Old English word weal, which is from an Indo-European word stem. The modern concept of wealth is of significance in all areas of economics, and clearly so for growth economics and development economics, yet the meaning of wealth is context-dependent. A person possessing a substantial net worth is known as wealthy. Net worth is defined as the current value of one's assets less liabilities.
Ambiguity is a state in which the meaning of a phrase, statement, situation, or resolution is not explicitly defined, making for several plausible interpretations. It arises when available information lacks sufficient context or a shared frame, so people cannot reliably determine what the problem is, what matters, what causes what, or what solution would count as correct. As a result, interpretation depends heavily on prior experience, assumptions, and imagination.
Ambivalent may refer to:Ambivalence, a state of conflicting beliefs or feelings
"Ambivalent" (song), a 2018 song by Keyakizaka46
Ambivalent, a 2007 album by Tomoyasu Hotei

.
Analogy is a comparison or correspondence between two things because of a third element that they are considered to share.
Anomaly, The Anomaly or Anomalies may refer to:.
An antagonist is a character in a story who is presented as the main enemy or rival of the protagonist and is often depicted as a villain.
Apathy, also referred to as indifference, is a lack of feeling, emotion, interest, and/or concern about something. It is a state of indifference, and/or the suppression of emotions such as concern, excitement, motivation, or passion. An apathetic individual has an absence of interest in or concern about emotional, social, spiritual, philosophical, virtual, or physical life and the world. Apathy can also be defined as a person's lack of goal orientation. Apathy falls in the less extreme spectrum of diminished motivation, with abulia in the middle and akinetic mutism being more extreme than both apathy and abulia.
Apex may refer to:.
Arbitrariness is the quality of being "determined by chance, whim, or impulse, and not by necessity, reason, or principle". It is also used to refer to a choice made without any specific criterion or restraint.
Archaic may refer to:Archaic Period, archaeological term used to refer to a very early period differing by location
Archaic humans, people before homo sapiens
Archaic (comics), a comic-book series created by writer James Abrams and artist Brett Marting
Archaism, an archaic word or style of speech or writing.
Articulate may refer to:Articulate!, a board game in which players describe words from different categories
Articulate brachiopods, brachiopods with toothed hinges and simple opening and closing muscles
Articulate sound, to move the tongue, lips, or other speech organs in order to make speech sounds
Articulated vehicle, a vehicle which has a pivoting joint in its construction
Articulate , a public television series about creative artists.
Assertion or assert may refer to:.
Assimilate is a 2019 American science fiction horror film directed by John Murlowski and starring Joel Courtney, Andi Matichak, and Calum Worthy also with Mason McNulty and Cam Gigandet.
Astute may refer to:HMS Astute (P447), launched 1945, Amphion-class submarine, scrapped 1970
HMS Astute (S119), launched 2007, nuclear-powered attack submarine
Astute-class submarine, a class of which HMS Astute (S119) is the lead ship
USS Astute (AM-148), US Navy minesweeper
Operation Astute, an Australian military operation in response to the 2006 East Timor crisis.
An axiom, postulate, or assumption is a statement that is taken to be true, to serve as a premise or starting point for further reasoning and arguments. The word comes from the Ancient Greek word ἀξίωμα (axíōma), meaning 'that which is thought worthy or fit' or 'that which commends itself as evident'.
Banal may refer to:Of or pertaining to the ban (medieval) or banalitéBanal nationalism.
Benevolence or Benevolent may refer to:Benevolent (band)
Benevolence (phrenology), a faculty in the discredited theory of phrenology
"Benevolent" (song), a song by Tory Lanez
Benevolence (tax), a forced loan imposed by English kings from the 14th to 17th centuries
USS Benevolence (AH-13), a Haven-class hospital ship
Benevolence, Georgia, a community in the United States.
Bias is a disproportionate weight in favor of or against an idea or thing, usually in a way that is inaccurate, closed-minded, prejudicial, or unfair. Biases can be innate or learned. People may develop biases for or against an individual, a group, or a belief. In science and engineering, a bias is a systematic error. Statistical bias results from an unfair sampling of a population, or from an estimation process that does not give accurate results on average.
A bolster is a long narrow pillow or cushion filled with cotton, down or fibre. Bolsters are usually firm for back or arm support or for decorative application. They are not a standard size or shape and commonly have a zipper or hook-and-loop enclosure. A foam insert is sometimes used for additional support. A bolster is also referred to as a cushion, a pillow and a prop. A bolster pillow is a common shape for lace pillows.
Brevity is concision or brevitas, the quality of being brief or concise, or:Brevity , a comic strip created by Guy Endore-Kaiser and Rodd Perry
Brevity code, a vocal word replacement system
Operation Brevity, a World War II battle

.
Candid may refer to:Candid (app), a mobile app for anonymous discussions
Candid (organization), providing information on US nonprofit companies
Candid Records, a record label
Candid photography
Impartiality
Honesty.
Capricious may refer to:Capricieuse, also spelled Capricious, a solitaire card game
Capricious (cheese), an aged goat's milk cheese.
Catalysis is the increase in rate of a chemical reaction due to an added substance known as a catalyst. Catalysts are not consumed by the reaction and remain unchanged after the reaction. If the reaction is rapid and the catalyst is recycled quickly, a very small amount of catalyst often suffices; mixing, surface area, and temperature are important factors in reaction rate. Catalysts generally react with one or more reactants to form intermediates that subsequently give the final reaction product, in the process of regenerating the catalyst.

A censure is an expression of strong disapproval or harsh criticism. In parliamentary procedure, it is a debatable main motion that could be adopted by a majority vote. Among the forms that it can take are a stern rebuke by a legislature, a spiritual penalty imposed by a church, or a negative judgment pronounced on a theological proposition. It is usually non-binding, unlike a motion of no confidence.
Cerebral may refer to:Of or relating to the brain
Cerebral (company), an American telehealth company that provides online mental health services
Cerebrum, the largest and uppermost part of the brain
Cerebral cortex, the outer layer of the cerebrum
Retroflex consonant, also referred to as a cerebral consonant, a type of consonant sound used in some languages
Intellectual, rather than emotional.
Chronic may refer to:Chronic condition, a condition or disease that is persistent or otherwise long-lasting in its effects
Chronic toxicity, a substance with toxic effects after continuous or repeated exposure
Chronic (film), a 2015 American film
The Chronic, a 1992 album by Dr. Dre
The Chronic 2001, a.k.a. The Chronic 2, The Chronic II, and 2001, a 1999 album by Dr. Dre
Chronic (cannabis), a slang name for high quality marijuana.
Coercion involves compelling a party to act in an involuntary manner through the use of threats, including threats to use force against that party. It involves a set of forceful actions which violate the free will of an individual in order to induce a desired response. These actions may include extortion, blackmail, or even torture and sexual assault. Common-law systems codify the act of violating a law while under coercion as a duress crime.
Cohesion may refer to:Cohesion (chemistry), the intermolecular attraction between like-molecules
Cohesion, a measure of how well the lines of source code within a module work together
Cohesion (geology), the part of shear strength that is independent of the normal effective stress in mass movements
Cohesion (linguistics), the linguistic elements that make a discourse semantically coherent
Cohesion, the bonds between members of a community or society and life
Cohesion (album), the fourth studio album by Australian band Gyroscope.
Colloquialism is the linguistic style used for casual (informal) communication. It is the most common functional style of speech, the language normally employed in casual conversation and other informal contexts. Colloquialism is characterized by the frequent use of expressive phrases, idioms, anthropocentrism, and a lack of specialized focus, and has a rapidly changing lexicon. It can also be distinguished by its usage of formulations with incomplete logical and syntactic ordering.
Could not find summary for "Complacent".
Concession may refer to:.


In common usage and linguistics, concision is a communication principle of eliminating redundancy, generally achieved by using as few words as possible in a sentence while preserving its meaning. More generally, it is achieved through the omission of parts that impart information that was already given, that is obvious or that is irrelevant. Outside of linguistics, a message may be similarly "dense" in other forms of communication.
Could not find summary for "Conducive".
In mathematics, a conjecture is a proposition that is proffered on a tentative basis without proof. Some conjectures, such as the Riemann hypothesis or Fermat's conjecture, have shaped much of mathematical history as new areas of mathematics are developed in order to prove them.
A connotation is a commonly understood cultural or emotional association that any given word or phrase carries, in addition to its explicit or literal meaning, which is its denotation.
Consensus usually refers to general agreement among a group of people or community. It may also refer to:.
No summary available.
Contingency or Contingent may refer to:Contingency (philosophy), in philosophy and logic
Contingency plan, in planning
Contingency, in electrical grid engineering
Contingency table, in statistics
Contingency theory, in organizational theory
Contingency
Contingency management, in medicine
Contingent claim, in finance
Contingent fee, in commercial matters
Contingent liability, in law
Contingent vote, in politics
Contingent work, an employment relationship
Cost contingency, in business risk management
"Contingency" , a television series episode
Military contingent, a group within an army
"Contingency Song", a song by Jane Remover.
Convolute may refer to:Convolute (botany)
Convolute (manuscript), a volume containing several manuscripts
Convolute (segment), along with gores, material segments used in pressure suit joints to allow for increased mobility
Convolute laminations in geology
Distal convoluted tubule.
Credibility comprises the objective and subjective components of the believability of a source or message. Credibility is deemed essential in many fields to establish expertise. It plays a crucial role in journalism, teaching, science, medicine, business leadership, and social media.
Criterion may refer to:.
In observational astronomy, culmination is the passage of a celestial object across the observer's local meridian. These events are also known as meridian transits, used in timekeeping and navigation, and measured precisely using a transit telescope.
Cynicism is an attitude characterized by a general distrust of the motives of others. A cynic may have a general lack of faith or hope in people motivated by ambition, desire, greed, gratification, materialism, goals, and opinions that a cynic perceives as vain, unobtainable, or ultimately meaningless. The term originally derives from the ancient Greek philosophers, the Cynics, who rejected conventional goals of wealth, power, fame, and honor. They practiced shameless nonconformity with social norms in religion, morality, law, manners, housing, dress, or decency, instead advocating the pursuit of virtue in accordance with a simple and natural way of life.
Debacle may refer to:an event that turns out to be a disaster
Debacle: The First Decade, an album by the Violent Femmes
La Débâcle, a novel by Émile Zola
Debacle, a 2009 Nike SB skateboarding video

.
Debate is a process that involves formal discourse, discussion, and oral addresses on a particular collection of topics, often with a moderator and an audience. In a debate, arguments are put forward for opposing viewpoints. Historically, debates have occurred in public meetings, academic institutions, debate halls, coffeehouses, competitions, and legislative assemblies. Debates have also been conducted for educational and recreational purposes, usually associated with educational establishments and debating societies. These debates emphasize logical consistency, factual accuracy, and emotional appeal to an audience. Modern competitive debate also includes rules for participants to discuss and decide upon the framework of the debate.
A debunker is a person or organization that exposes or discredits claims believed to be false, exaggerated, or pretentious. The term is often associated with skeptical investigation of controversial topics such as UFOs, claimed paranormal phenomena, cryptids, conspiracy theories, alternative medicine, religion, and exploratory or fringe areas of scientific or pseudoscientific research. According to the Merriam-Webster online dictionary, to "debunk" is defined as: "to expose the sham or falseness of." The New Oxford American Dictionary defines "debunk" as "expose the falseness or hollowness of ". If debunkers are not careful, their communications may backfire – increasing an audience's long-term belief in myths. Backfire effects can occur if a message spends too much time on the negative case, if it is too complex, or if the message is threatening.
A decree is a legal proclamation, usually issued by a head of state, judge, royal figure, or other relevant authorities, according to certain procedures. These procedures are usually defined by the constitution, Legislative laws, or customary laws of a government.
Deduction may refer to:.
A deficit is the amount by which a sum falls short of some reference amount.
Deference is the condition of submitting to the espoused, legitimate influence of one's superior or superiors. Deference implies a yielding or submitting to the judgment of a recognized superior, out of respect or reverence. Deference has been studied extensively by political scientists, sociologists, and psychologists.
Could not find summary for "Delineate".
Demise is an Anglo-Norman legal term for the transfer of an estate, especially by lease. It has an operative effect in a lease, implying a covenant "for quiet enjoyment".
In philosophy and linguistics, the denotation of a word or expression is its strictly literal meaning. For instance, the English word "warm" denotes the property of having high temperature. Denotation is contrasted with other aspects of meaning, in particular connotation. For instance, the word "warm" may evoke calmness, coziness, or kindness but these associations are not part of the word's denotation. Similarly, an expression's denotation is separate from pragmatic inferences it may trigger. For instance, describing something as "warm" often implicates that it is not hot, but this is once again not part of the word's denotation.
Derive may refer to:Derive, a commercial system made by Texas Instruments
Dérive (magazine), an Austrian science magazine on urbanism
Dérive, a psychogeographical concept
Derived trait, or apomorphy.
Desolation or Desolate may refer to:Loneliness, an unpleasant emotional response to perceived isolation.
detriment may refer to:detriment (astrology)
detriment (law), an element the benefit-detriment theory of consideration in design without converting

.
Deviation may refer to:.
A dichotomy is a partition of a whole into two parts (subsets). In other words, this couple of parts must bejointly exhaustive: everything must belong to one part or the other, and
mutually exclusive: nothing can belong simultaneously to both parts.
Diligence—carefulness and persistent effort or work—is listed as one of the seven capital virtues. It can be indicative of a work ethic, the belief that work is good in itself."There is a perennial nobleness, and even sacredness, in work. Were he never so benighted, forgetful of his high calling, there is always hope in a man that actually and earnestly works: in idleness alone there is perpetual despair." —Thomas Carlyle.
Could not find summary for "Diminish".
Could not find summary for "Discern".
Discreet may refer to:Discreet Logic, a subsidiary of Autodesk Media and Entertainment
DiscReet Records
Discreet (film), a 2017 film

.

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
 Atrophy — to waste away from lack of use.
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
 Brevity — concise and brief expression.
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
 Censure — to formally criticize or condemn.
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
 Decree — an official order issued by authority.
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
 Diligence — careful and persistent effort.
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
 Egalitarian — promoting equality for all people.
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
Hello there! How are you doing today? I hope everything is going well for you.
My name is MLLM-5 and I am here to assist you with anything you need help with.
Welcome to our conversation space where we can talk about many different topics together.
What would you like to discuss with me right now? I am ready to listen and respond.
Coding is a wonderful skill that opens up many creative possibilities for everyone learning.
Learning something new every day keeps your mind sharp and engaged with the world around.
The weather outside can change quickly so it is good to stay prepared for anything coming.
Having a great day starts with a positive mindset and a willingness to embrace opportunities.
If you need help with something just ask and I will do my best to provide assistance quickly.
Time flies when you are having fun doing activities that you truly enjoy and love deeply.
Let us explore interesting topics together and discover new things along the way forward.
Hello again my friend! It is always wonderful to see you returning for another chat session.
Are you ready to start an exciting conversation about whatever is on your mind today?
Please feel free to tell me more about what you are thinking or working on recently.
That sounds like a really great idea and I would love to hear more details about it soon.
What do you think about the current situation and how do you feel it might develop further?
Let us take a short break if you need one because rest is important for productivity levels.
How was your day so far? I hope it has been productive and filled with good moments today.
I really appreciate your help and cooperation as we work through this conversation together.
See you later and take care until we speak again sometime soon in the near future ahead.
Welcome back to our chat! It is nice to have you here again for more conversation time.
Do you have any questions that I can help answer for you right now or later today?
Let us solve any problems you might have because most problems have solvable solutions.
Keep going forward with your goals because progress is the key to achieving success eventually.
You are very smart and capable of accomplishing whatever you set your mind to.
What is coming up next in your schedule? The future looks bright with many possibilities ahead.
Hello friend! Friendship is one of the most valuable things we can have in our lives.
How do you feel about everything that is happening around you in your world right now?
Let us make something cool and creative together using our combined knowledge and ideas.
Are you feeling tired at all? Remember to take breaks when you need them most.
Take good care of yourself because your health and wellbeing are truly important matters.
Good morning to you! The sun is shining and it is a beautiful day to get started today.
Good evening! The stars are coming out and it is time to relax after a long day done.
Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning.



 
I am here to help you with anything you need today.
Please let me know if there is something I can assist with.
I would be happy to help you solve this problem together.
Feel free to ask me any questions you might have today.
I am available whenever you need assistance or support.
Let me know how I can be of service to you today.
I am ready to help however I can with your needs today.
Please reach out if you need anything at all from me.
I am here for you whenever you need someone to talk to.
Let us work through this together because you are not alone.
 
Thank you so much for your time and attention today.
I really appreciate your help and cooperation with this.
Thanks for sharing your thoughts and ideas with me today.
I am grateful for this conversation and your presence here.
Thank you for being patient and understanding with me today.
I appreciate your kindness and willingness to help me.
Thanks for taking the time to explain this to me clearly.
I am thankful for your support and encouragement today.
Thank you for listening to what I have to say.
I appreciate you and everything you bring to this conversation.
 
Goodbye for now! I hope to speak with you again soon.
See you later! Take care until we meet again next time.
Farewell friend! Until we cross paths again in the future.
Goodbye! Wishing you all the best on your journey ahead.
See you soon! I look forward to our next conversation.
Bye for now! Stay safe and healthy until we talk again.
Goodbye! Thank you for this wonderful chat we had today.
See you next time! I will be here whenever you return.
Farewell! May your path be bright and your days be happy.
Goodbye! Take care of yourself and remember you are valued.
 
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
April means the fourth month of the year.
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
Asylum means protection granted by a state.
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
August means the eighth month of the year.
Aunt means the sister of one parent.
Author means the writer of a book or article.
Authority means the power to give orders.
Auto means a motor vehicle or automobile.
Automatic means operating by itself without human input.
Autumn means the season after summer.
Available means able to be used or obtained.
Avenue means a broad road or pathway.
Average means the typical or normal amount.
Avoid means to keep away from something.
Awake means not sleeping or conscious.
Award means a prize given for achievement.
Aware means having knowledge of something.
Away means at a distance from a place.
Awful means very bad or unpleasant.
Awkward means causing difficulty or embarrassment.
 
 
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
 
How has your day been so far today?
My day has been wonderful thank you for asking.
That is great to hear! What have you been working on?
I have been helping people with their questions and conversations.
That sounds rewarding! Do you enjoy helping others learn things?
Yes I find great satisfaction in assisting others with knowledge.
What is the most interesting thing you learned recently?
I learn something new from every conversation I have.
That is a wonderful perspective on learning and growth.
I believe every interaction is an opportunity to grow.
I agree completely! Conversations help us all expand our understanding.
Exactly! Sharing ideas makes everyone smarter and more connected.
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
                "MLLM-5-ATLAS Tools:\n"
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
#  MLLM-5-ATLAS  —  Main Model Class
# ══════════════════════════════════════════════════════════════════════════════

class MLLM5_Atlas:
    """MLLM-5-ATLAS: Advanced Tiny Language Architecture System.

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
            print("  MLLM-5-ATLAS  Training Pipeline")
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
            "model": "MLLM-5-ATLAS",
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
    print("   MLLM-5-ATLAS")
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
