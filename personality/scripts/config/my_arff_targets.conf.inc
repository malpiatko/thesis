//
// configuration of commandline options for target classes in an ARFF file
// (cArffSink)
//
// change this file to match your labels!
//

class[0].name = \cm[na1{Speaker_ID}: attribute name]
class[0].type = \cm[cl1{numeric}: all classes for arff file attribute]
target[0].all = \cm[Speaker_ID{unknown}: instance class label]

class[1].name = \cm[na2{Gender}: attribute name]
class[1].type = {M,F}
target[1].all = \cm[Gender{unknown}: instance class label]

class[2].name = \cm[na3{Status}: attribute name]
class[2].type = {J,G}
target[2].all = \cm[Status{unknown}: instance class label]

class[3].name = \cm[na4{Extraversion}: attribute name]
class[3].type = {0,1}
target[3].all = \cm[Extraversion{unknown}: instance class label]

class[4].name = \cm[na5{Agreeableness}: attribute name]
class[4].type = {0,1}
target[4].all = \cm[Agreeableness{unknown}: instance class label]

class[5].name = \cm[na6{Conscientiousness}: attribute name]
class[5].type = {0,1}
target[5].all = \cm[Conscientiousness{unknown}: instance class label]

class[6].name = \cm[na7{Neuroticism}: attribute name}
class[6].type = {0,1}
target[6].all = \cm[Neuroticism{unknown}: instance class label]

class[7].name = \cm[na8{Openness}: attribute name]
class[7].type = {0,1}
target[7].all = \cm[Openness{unknown}: instance class label]



// use this for a nominal label (discrete classes)
// class[0].type = {classA,classB,classC}  
//class[0].type = \cm[classtype{numeric}:type of labelA, either numeric or a list of comma separated classes in curly brackets]
//target[0].all = \cm[class{?}:string value for labelA]

/*
class[1].name = labelB
class[0].type = \cm[labelBtype{numeric}:type of labelB, either numeric or a list of comma separated classes in curly brackets]
target[1].all = \cm[labelB{0.0}:numeric value for labelB]
// you can add as many labels here as you want
*/

