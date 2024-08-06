import styles from '~/styles/Dashboard.css'
import React, {useEffect, useState} from "react";
import {useDisclosure} from "@mantine/hooks";
import {
    ActionIcon,
    AppShell,
    Burger, Button, Center, Highlight,
    Flex, Group, LoadingOverlay, Modal, NumberInput, Pagination, rem, Select, Slider, Stepper, Switch,
    Text, useComputedColorScheme, useMantineColorScheme,
} from "@mantine/core";
import {Outlet, useNavigate, useOutletContext} from "@remix-run/react";

import {LinksFunction} from "@remix-run/node";
import {
    IconCheck,
    IconCpu,
    IconMessage,
    IconMoon,
    IconSearch,
    IconSelector,
    IconSitemap,
    IconSun, IconX
} from "@tabler/icons-react";
import {notifications} from "@mantine/notifications";
import {Modality} from "~/pojo/modality";
import {encoders} from "~/pojo/encoder";
import {Dataset, datasets} from "~/pojo/dataset";

export const links: LinksFunction = () => [
    {rel: "stylesheet", href: styles},
];

const algorithms = ["Flat", "KGraph", "NSG", "NSSG", "Vamana"]
const LLMs = ['none', 'gpt-3.5-turbo', 'gpt-4-turbo', 'dall-e-3']

type ContextType = {
    stepper: number,
    llm: string,
    useKnowledge: boolean,
    temperature: number,
    resultNumber: number,
    algorithm: string,
    neighbor: number,
    candidate: number,
    indexWeight: number[],
    retrievalFramework: string,
    retrievalWeight: number[]
};

export default function DashBoard() {
    // Color scheme toggle
    const {setColorScheme} = useMantineColorScheme();
    const computedColorScheme = useComputedColorScheme('light', {getInitialValueInEffect: true});

    const navigate = useNavigate();
    const [opened, {toggle}] = useDisclosure();

    // Stepper
    const [stepper, setStepper] = useState<number>(0);
    const [stepperDescription1, setStepperDescription1] = useState<string>(
        `Knowledge: |# Objects: |Modal:`
    );
    const [stepperDescription2, setStepperDescription2] = useState<string>(
        'Encoder: |Dimension: |# Vector:'
    );
    const [stepperDescription3, setStepperDescription3] = useState<string>(
        `Index Type: ${algorithms.map((_, index) => (index == 0 ? '' : '\n') + _)}|Framework:`
    );
    const [stepperDescription4, setStepperDescription4] = useState<string>(
        'Framework: |# Results:'
    );
    const [stepperDescription5, setStepperDescription5] = useState<string>(
        'Modal: |Temperature: '
    );

    // Header & three setting Modals
    const [showKnowledgeModal, setShowKnowledgeModal] = useState(false);
    const [showEmbeddingModal, setShowEmbeddingModal] = useState(false);
    const [showIndexModal, setShowIndexModal] = useState(false);
    const [showRetrievalModal, setShowRetrievalModal] = useState(false);
    const [showLlmModal, setShowLlmModal] = useState(false);

    // Knowledge settings
    const [useKnowledge, setUseKnowledge] = useState<boolean>(false);
    const [dataset, setDataset] = useState<Dataset>(datasets[0]);

    // Embedding settings
    const [encoderNumber, setEncoderNumber] = useState<number>(1);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [loading, setLoading] = useState<boolean>(false);
    const [activePage, setPage] = useState(1);
    const [modalities, setModalities] = useState<Modality[]>([])
    const [learning, setLearning] = useState<boolean>(false);

    const postEmbedding = async () => {
        setLoading(true);
        console.log(modalities);
        const requestBody = {
            'modalities': modalities.map((_) => ({
                encoder: parseInt(_.encoder?.id as string),
                modals: _.modals?.map((__) => {
                    return parseInt(__.id);
                })
            })),
            'learning': learning
        }
        console.log(requestBody)
        const id = notifications.show({
            loading: true,
            title: 'Processing Embedding',
            message: 'Data is still processing, you can check the process from the backend',
            autoClose: false,
            withCloseButton: false,
        });

        // toggle();
        const response = await fetch('http://127.0.0.1:4523/m1/4132394-0-default/embedding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        const data = await response.json();
        setLoading(false);

        if (data.message) {
            notifications.update({
                id,
                color: 'red',
                title: 'Failed!',
                message: data.message,
                icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                loading: false,
                withCloseButton: true,
            })
        } else {
            notifications.update({
                id,
                color: 'teal',
                title: 'Success!',
                message: 'Embedding configuration set successfully',
                icon: <IconCheck style={{width: rem(18), height: rem(18)}}/>,
                loading: false,
                withCloseButton: true,
                autoClose: 5000,
            });
            setIndexWeight(data.data.weight)
            setRetrievalWeight(data.data.weight)
            // setIndexWeight(modalities.map((_) => _.weight!))
            // setRetrievalWeight(modalities.map((_) => _.weight!))
            setStepper(2);
            setStepperDescription2(`Encoder:${modalities.map((_) => '\n' + _.encoder!.name)}| Dimension: ${modalities.map((_) => '\n' + _.encoder!.dimension.toString())}|# vector: ${encoderNumber}`);
        }
    }

    // Index settings
    const [algorithm, setAlgorithm] = useState<string>('Flat');
    const [neighbor, setNeighbor] = useState<number>(30);
    const [candidate, setCandidate] = useState<number>(200);
    const [indexWeight, setIndexWeight] = useState<number[]>([]);
    const [indexFramework, setIndexFramework] = useState<string>('MUST');
    const [createdIndex, setCreatedIndex] = useState<Record<string, boolean>>(
        algorithms.reduce((acc, algorithm) => {
            return {...acc, [algorithm]: false};
        }, {})
    );

    const postIndex = async () => {
        setLoading(true);
        console.log(modalities);
        const requestBody = {
            'algorithm': algorithm,
            'neighbor': neighbor,
            'candidate': candidate,
            'index_weight': indexWeight
        }
        console.log(requestBody)
        const id = notifications.show({
            loading: true,
            title: 'Processing Create Index',
            message: 'Data is still processing...',
            autoClose: false,
            withCloseButton: false,
        });

        // toggle();
        const response = await fetch('http://127.0.0.1:4523/m1/4132394-0-default/index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        const data = await response.json();
        setLoading(false);

        if (data.message) {
            notifications.update({
                id,
                color: 'red',
                title: 'Failed!',
                message: data.message,
                icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                loading: false,
                withCloseButton: true,
            })
        } else {
            notifications.update({
                id,
                color: 'teal',
                title: 'Success!',
                message: 'Index configuration set successfully',
                icon: <IconCheck style={{width: rem(18), height: rem(18)}}/>,
                loading: false,
                withCloseButton: true,
                autoClose: 5000,
            });
            const newCreatedIndex = {...createdIndex};
            newCreatedIndex[algorithm] = true;
            setCreatedIndex(newCreatedIndex);
            if (stepper < 3) setStepper(3);
            setStepperDescription3(`Index Type: ${algorithms.map((_, index) => (index == 0 ? '' : '\n') + _)}|Framework: ${indexFramework}`)
        }
    }

    // Retrieval settings
    const [resultNumber, setResultNumber] = useState<number>(3);
    const [retrievalWeight, setRetrievalWeight] = useState<number[]>([]);
    const [retrievalFramework, setRetrievalFramework] = useState<string>('MUST');

    // LLM settings
    const [llm, setLlm] = useState<string>('gpt-3.5-turbo');
    const [temperature, setTemperature] = useState<number>(0.5);
    // if not use knowledge base, then use llm
    useEffect(() => {
        if (!useKnowledge && llm == 'none') {
            setLlm(LLMs[1]);
            notifications.show({
                title: 'LLM model changed',
                message: 'LLM has been automatically changed to GPT-3.5-turbo due to the unused knowledge base.'
            })
        }
    }, [llm, useKnowledge])

    return (
        <div>
            <AppShell
                layout="alt"
                header={{height: 60}}
                navbar={{width: 250, breakpoint: 'sm', collapsed: {mobile: !opened},}}
                padding="md"
            >
                <AppShell.Header>
                    <Flex justify="space-between" align='center' h='100%' ml={15} mr={15} className={"inner"}>
                        <Burger opened={opened}
                                onClick={toggle}
                                hiddenFrom="sm"
                                size="sm"/>
                        <Text
                            size={'xl'}
                            fw={900}
                            style={{fontFamily: 'Arial'}}
                            onClick={() => {
                                navigate('/about')
                            }}
                        >MQA</Text>
                        <Group gap={5}>
                            <Button
                                onClick={() => {
                                    setShowKnowledgeModal(true);
                                }}
                                className={"link"}
                            >Knowledge Base</Button>
                            <Button
                                onClick={() => {
                                    if (!useKnowledge) {
                                        notifications.show({
                                            color: 'red',
                                            icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                                            title: 'Embedding Configuration Error',
                                            message: 'Please set "Use Knowledge Base" and select a dataset first',
                                            autoClose: 5000,
                                            withCloseButton: true
                                        });
                                        return;
                                    }
                                    setShowEmbeddingModal(true);
                                }}
                                className={"link"}
                            >Embedding</Button>
                            <Button
                                onClick={() => {
                                    setShowIndexModal(true);
                                }}
                                className={"link"}
                            >Index</Button>
                            <Button
                                onClick={() => {
                                    setShowRetrievalModal(true);
                                }}
                                className={"link"}
                            >Retrieval</Button>
                            <Button
                                onClick={() => {
                                    setShowLlmModal(true);
                                }}
                                className={"link"}
                            >LLM</Button>
                        </Group>
                        <Group justify="center">
                            <ActionIcon
                                onClick={() => setColorScheme(computedColorScheme === 'light' ? 'dark' : 'light')}
                                variant="default"
                                size="xl"
                                aria-label="Toggle color scheme"
                            >
                                {computedColorScheme === 'light' ? <IconMoon/> : <IconSun/>}
                            </ActionIcon>
                        </Group>
                    </Flex>
                </AppShell.Header>
                <AppShell.Navbar p="md">
                    <Group justify={'start'}>
                        <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm"/>
                        {/*<Text>Timeline</Text>*/}
                    </Group>
                    <Flex
                        h="100%"
                        direction="column"
                        align="center"
                    >
                        <Center h='100%'>
                            <Stepper active={stepper} orientation="vertical" styles={{
                                stepLabel: {fontSize: rem(20)},
                                step: {minHeight: rem(100)},
                                stepDescription: {
                                    whiteSpace: 'pre-line', fontSize: rem(18), lineHeight: rem(20)
                                }
                            }}>
                                <Stepper.Step className={'stepDescription'} icon={<IconSelector size={20}/>}
                                              label="Data Preprocessing"
                                              description={
                                                  <Text>
                                                      {stepperDescription1.split('|').map(_ => {
                                                          const __ = _.split(':')
                                                          return (<React.Fragment key={__[0]}>
                                                              • {__[0]}: {<Text span c="pink"
                                                                                inherit>{__[1] + "\n"}</Text>}
                                                          </React.Fragment>);
                                                      })}
                                                  </Text>
                                              }/>
                                <Stepper.Step icon={<IconCpu size={20}/>} label="Vector Representation"
                                              description={
                                                  <Text>
                                                      {stepperDescription2.split('|').map(_ => {
                                                          const __ = _.split(':')
                                                          return (<React.Fragment key={__[0]}>
                                                              • {__[0]}: {<Text span c="pink"
                                                                                inherit>{__[1] + "\n"}</Text>}
                                                          </React.Fragment>);
                                                      })}
                                                  </Text>
                                              }/>
                                <Stepper.Step icon={<IconSitemap size={20}/>} label="Index Construction"
                                              description={
                                                  <Text>
                                                      {stepperDescription3.split('|').map((_, index) => {
                                                          const __ = _.split(':')
                                                          if (index == 0) {
                                                              return (<React.Fragment key={__[0]}>
                                                                  • {__[0]}:
                                                                  <Highlight
                                                                      highlight={algorithms.filter(_ => createdIndex[_])}
                                                                      highlightStyles={{
                                                                          backgroundImage:
                                                                              'linear-gradient(45deg, var(--mantine-color-cyan-5), var(--mantine-color-indigo-5))',
                                                                          fontWeight: 700,
                                                                          WebkitBackgroundClip: 'text',
                                                                          WebkitTextFillColor: 'transparent',
                                                                      }}
                                                                  >{__[1]}</Highlight>
                                                              </React.Fragment>);
                                                          } else {
                                                              return (<React.Fragment key={__[0]}>
                                                                  • {__[0]}: {<Text span c="pink"
                                                                                    inherit>{__[1] + "\n"}</Text>}
                                                              </React.Fragment>);
                                                          }
                                                      })}
                                                  </Text>
                                              }/>
                                <Stepper.Step icon={<IconSearch size={20}/>} label="Query Execution"
                                              description={
                                                  <Text>
                                                      {stepperDescription4.split('|').map(_ => {
                                                          const __ = _.split(':')
                                                          return (<React.Fragment key={__[0]}>
                                                              • {__[0]}: {<Text span c="pink"
                                                                                inherit>{__[1] + "\n"}</Text>}
                                                          </React.Fragment>);
                                                      })}
                                                  </Text>
                                              }/>
                                <Stepper.Step icon={<IconMessage size={20}/>} label="Answer Generation"
                                              description={
                                                  <Text>
                                                      {stepperDescription5.split('|').map(_ => {
                                                          const __ = _.split(':')
                                                          return (<React.Fragment key={__[0]}>
                                                              • {__[0]}: {<Text span c="pink"
                                                                                inherit>{__[1] + "\n"}</Text>}
                                                          </React.Fragment>);
                                                      })}
                                                  </Text>
                                              }/>
                            </Stepper>
                        </Center>
                    </Flex>
                </AppShell.Navbar>
                <AppShell.Main>
                    <Outlet context={{
                        stepper,
                        llm,
                        useKnowledge,
                        temperature,
                        resultNumber,
                        algorithm,
                        neighbor,
                        candidate,
                        indexWeight,
                        retrievalFramework,
                        retrievalWeight
                    } satisfies ContextType}/>
                </AppShell.Main>
            </AppShell>

            <Modal opened={showKnowledgeModal}
                   onClose={() => {
                       setShowKnowledgeModal(false)
                   }}
                   title="Knowledge Base Configuration"
                   overlayProps={{
                       backgroundOpacity: 0.55,
                       blur: 3,
                   }}
            >
                <Switch label={"Use Knowledge"} checked={useKnowledge}
                        onChange={(event) => {
                            setUseKnowledge(event.currentTarget.checked);
                            if (dataset) {
                                setStepperDescription1(`Knowledge: ${dataset.name}|# Objects: ${dataset.objects}|Modal: ${dataset.modal.map(_ => _.type)}`)
                            }
                        }}/>

                <Select
                    label={"Select Dataset"}
                    data={datasets.map(value => value.name)}
                    value={dataset.name}
                    mt={'md'}
                    onChange={(name) => {
                        if (name == null) return;
                        setDataset(datasets.find(_ => _.name == name) || datasets[0])
                        setStepperDescription1(`Knowledge: ${dataset.name}|Objects: ${dataset.objects}|# Modal: ${dataset.modal.map(_ => _.type)}`)
                    }}
                    placeholder={dataset.name}
                    allowDeselect={false}
                    disabled={!useKnowledge}
                />

                <Group justify={'flex-end'} mt={'md'}>
                    <Button style={{
                        background: 'var(--mantine-color-red-9)',
                        hover: 'var(--mantine-color-red-8)',
                        color: 'var(--mantine-color-white)',
                        border: 'none',
                    }}
                            onClick={() => {
                                setUseKnowledge(false);
                                setStepper(1);
                            }}
                    >
                        Reset
                    </Button>
                    <Button variant="gradient"
                            gradient={{from: 'blue', to: 'cyan', deg: 150}}
                            onClick={() => {
                                notifications.show({
                                    color: 'teal',
                                    title: 'Success!',
                                    message: 'Knowledge base configuration set successfully',
                                    icon: <IconCheck style={{width: rem(18), height: rem(18)}}/>,
                                    loading: false,
                                    withCloseButton: true,
                                    autoClose: 5000,
                                })
                                setStepper(useKnowledge ? 1 : 0);
                                setShowKnowledgeModal(false)
                            }}
                    >
                        Submit
                    </Button>
                </Group>
            </Modal>

            <Modal
                opened={showEmbeddingModal}
                onClose={() => {
                    setShowEmbeddingModal(false)
                }}
                title="Embedding Configuration"
                overlayProps={{
                    backgroundOpacity: 0.55,
                    blur: 3,
                }}
            >
                <LoadingOverlay visible={loading} loaderProps={{children: 'Loading...'}}/>
                {currentPage === 1 &&
                    <>
                        <NumberInput
                            label="Set Modal Number(m)"
                            value={encoderNumber}
                            onChange={(value: string | number) => {
                                if (value == null) return;
                                const val = typeof value === 'number' ? value : parseInt(value);
                                setEncoderNumber(val);
                            }}
                            min={1}
                            allowDecimal={false}
                        />
                        <Switch label={"Process Weight Learning"} checked={learning} mt={'md'}
                                onChange={(event) => {
                                    setLearning(event.currentTarget.checked);
                                }}/>
                        <Flex justify={'flex-end'} mt={'md'}>
                            <Button disabled={!dataset}
                                    onClick={() => {
                                        setModalities(Array.from({length: encoderNumber}, () => {
                                            return {
                                                encoder: undefined,
                                                modals: undefined,
                                                weight: 1 / encoderNumber,
                                            };
                                        }));
                                        setPage(1);
                                        setCurrentPage((prevPage) => prevPage + 1)
                                    }}>
                                Next
                            </Button>
                        </Flex>
                    </>}
                {currentPage === 2 &&
                    <>
                        <Center><Pagination total={modalities.length} value={activePage} onChange={setPage}/></Center>
                        {modalities.length > 0 &&
                            <>
                                <Select
                                    key={activePage}
                                    label={"Choose Encoder"}
                                    data={encoders.map((encoder) => ({
                                        value: encoder.id,
                                        label: encoder.name
                                    }))}
                                    value={modalities[activePage - 1].encoder ? modalities[activePage - 1].encoder?.id : null}
                                    onChange={(value) => {
                                        if (value == null) return;
                                        console.log(value);
                                        const newModalities = [...modalities];
                                        const encoder = encoders.find((_) => _.id === value);
                                        newModalities[activePage - 1].encoder = encoder;
                                        // the encoder must exist
                                        newModalities[activePage - 1].modals = Array.from({length: encoder!.type.length}, () => ({
                                            id: '',
                                            name: '',
                                            type: ''
                                        }));
                                        setModalities(newModalities);
                                    }}
                                    placeholder="Select Encoder"
                                    mt={10}
                                    allowDeselect={false}
                                />
                                {modalities[activePage - 1].encoder?.type.map((type, index) => (
                                    <Select
                                        key={index}
                                        label={`Choose ${type} Modal`}
                                        data={dataset.modal.filter((modal) => modal.type === type).map((modal) => ({
                                            value: modal.id,
                                            label: modal.name
                                        }))}
                                        // the modals must not be empty
                                        value={modalities[activePage - 1].modals![index]?.id}
                                        onChange={(value) => {
                                            if (value == null) return;
                                            const newModalities = [...modalities];
                                            const modal = dataset.modal.find((_) => _.id == value);
                                            if (modal == undefined) return;
                                            newModalities[activePage - 1].modals![index] = modal;
                                            setModalities(newModalities);
                                        }}
                                        mt={10}
                                        allowDeselect={false}
                                    />
                                ))}
                            </>
                        }
                        <Group justify={'flex-end'} gap={'md'} mt={'md'}>
                            <Button onClick={() => {
                                setCurrentPage((prevPage) => prevPage - 1)
                            }}>
                                Prev
                            </Button>
                            <Button onClick={() => {
                                let flag = true;
                                for (const modality of Object.values(modalities)) {
                                    if (!modality.encoder || !modality.modals || !modality.weight) {
                                        flag = false;
                                        break;
                                    }
                                    for (const modal of Object.values(modality.modals)) {
                                        if (!modal.id || !modal.type || !modal.name) {
                                            flag = false;
                                            break;
                                        }
                                    }
                                    if (!flag) break;
                                }
                                if (!flag) {
                                    notifications.show({
                                        color: 'red',
                                        icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                                        autoClose: 5000,
                                        withCloseButton: true,
                                        title: 'Embedding Configuration Error',
                                        message: 'Please select a value for all fields',
                                    });
                                    return;
                                }
                                setShowEmbeddingModal(false);
                                setCreatedIndex(algorithms.reduce((acc, algorithm) => {
                                    return {...acc, [algorithm]: false};
                                }, {}));
                                postEmbedding()
                            }}>
                                Submit
                            </Button>
                        </Group>
                    </>}
            </Modal>

            <Modal opened={showIndexModal} onClose={() => {
                setShowIndexModal(false)
            }}
                   title={"Index Configuration"}
                   overlayProps={{
                       backgroundOpacity: 0.55,
                       blur: 3,
                   }}
            >
                <Text>Index Methods</Text>
                <Select
                    data={algorithms}
                    value={algorithm} onChange={(value) => {
                    setAlgorithm(value!)
                }}
                    allowDeselect={false}/>
                <Text mt={'md'}>Index Framework</Text>
                <Select
                    data={["MUST", "MR", "JE"]}
                    value={indexFramework} onChange={(value) => {
                    setIndexFramework(value!)
                }}
                    allowDeselect={false}/>
                <Text mt={'md'}>Neighbor</Text>
                <NumberInput
                    defaultValue={30}
                    min={1}
                    value={neighbor}
                    onChange={(value) => {
                        setNeighbor(typeof value == 'string' ? parseInt(value) : value)
                    }}
                    allowDecimal={false}
                />
                <Text mt={'md'}>Candidate</Text>
                <NumberInput
                    defaultValue={30}
                    min={1}
                    value={candidate}
                    onChange={(value) => {
                        setCandidate(typeof value == 'string' ? parseInt(value) : value)
                    }}
                    allowDecimal={false}
                />
                {indexWeight.map((value, index) => (
                    <div key={index}>
                        <Text mt={'md'}>Modal {index} Weight</Text>
                        <NumberInput
                            key={index}
                            value={value}
                            onChange={(newValue) => {
                                if (newValue != null) {
                                    setIndexWeight((prevIndexWeight) => {
                                        const newValues = [...prevIndexWeight];
                                        newValues[index] = typeof newValue === 'string' ? parseFloat(newValue) : newValue;
                                        return newValues;
                                    });
                                }
                            }}
                            decimalScale={2}
                        />
                    </div>
                ))}
                <Group justify={'flex-end'} mt={'md'}>
                    <Button style={{
                        background: 'var(--mantine-color-red-9)',
                        hover: 'var(--mantine-color-red-8)',
                        color: 'var(--mantine-color-white)',
                        border: 'none',
                    }}
                            onClick={() => {
                                setAlgorithm('Flat');
                                setNeighbor(30);
                                setCandidate(200);
                                setIndexWeight(Array.from({length: modalities.length}, () => 1));
                            }}
                    >
                        Reset
                    </Button>
                    <Button variant="gradient"
                            gradient={{from: 'blue', to: 'cyan', deg: 150}}
                            onClick={() => {
                                if (!algorithm || !neighbor || !candidate || !indexWeight) {
                                    notifications.show({
                                        color: 'red',
                                        withCloseButton: true,
                                        title: 'Index Configuration Error',
                                        message: 'Please select a value for all fields',
                                        icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                                        autoClose: 5000,
                                    });
                                    return;
                                }
                                if (candidate < neighbor) {
                                    notifications.show({
                                        color: 'red',
                                        withCloseButton: true,
                                        title: 'Index Configuration Error',
                                        message: '"Candidate" must greater or equal to "Neighbor"',
                                        icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                                        autoClose: 5000,
                                    });
                                    return;
                                }
                                setShowIndexModal(false);
                                postIndex();
                            }}
                    >
                        Submit
                    </Button>
                </Group>
            </Modal>

            <Modal opened={showRetrievalModal} onClose={() => {
                setShowRetrievalModal(false)
            }}
                   title={"Retrieval Configuration"}
                   overlayProps={{
                       backgroundOpacity: 0.55,
                       blur: 3,
                   }}
            >
                <Text>Retrieval Framework</Text>
                <Select
                    data={["MUST", "MR", "JE"]}
                    value={retrievalFramework} onChange={(value) => {
                    setRetrievalFramework(value!)
                }}
                    allowDeselect={false}/>
                <Text mt={'md'}>Retrieval Number</Text>
                <NumberInput
                    defaultValue={3}
                    min={1}
                    value={resultNumber}
                    onChange={(value) => {
                        setResultNumber(typeof value == 'string' ? parseInt(value) : value)
                    }}
                    allowDecimal={false}
                />
                {retrievalWeight.map((value, index) => (
                    <div key={index}>
                        <Text mt={'md'}>Modal {index} Weight</Text>
                        <NumberInput
                            key={index}
                            value={value}
                            onChange={(newValue) => {
                                if (newValue != null) {
                                    setRetrievalWeight((prevRetrievalWeight) => {
                                        const newValues = [...prevRetrievalWeight];
                                        newValues[index] = typeof newValue === 'string' ? parseFloat(newValue) : newValue;
                                        return newValues;
                                    });
                                }
                            }}
                            decimalScale={2}
                        />
                    </div>
                ))}
                <Group justify={'flex-end'} mt={'md'}>
                    <Button style={{
                        background: 'var(--mantine-color-red-9)',
                        hover: 'var(--mantine-color-red-8)',
                        color: 'var(--mantine-color-white)',
                        border: 'none',
                    }}
                            onClick={() => {
                                setResultNumber(3);
                                setRetrievalWeight(Array.from({length: modalities.length}, () => 1))
                            }}
                    >
                        Reset
                    </Button>
                    <Button variant="gradient"
                            gradient={{from: 'blue', to: 'cyan', deg: 150}}
                            onClick={() => {
                                setShowRetrievalModal(false);
                                if (stepper < 4) setStepper(4);
                                setStepperDescription4(`Framework: ${retrievalFramework}|# Results: ${resultNumber}`)
                                notifications.show({
                                    color: 'teal',
                                    title: 'Success!',
                                    message: 'Retrieval configuration set successfully',
                                    icon: <IconCheck style={{width: rem(18), height: rem(18)}}/>,
                                    loading: false,
                                    withCloseButton: true,
                                    autoClose: 5000,
                                })
                            }}
                    >
                        Submit
                    </Button>
                </Group>
            </Modal>
            <Modal opened={showLlmModal}
                   onClose={() => {
                       setShowLlmModal(false)
                   }}
                   title="LLM configuration"
                   overlayProps={{
                       backgroundOpacity: 0.55,
                       blur: 3,
                   }}
            >
                <Text>LLM Model</Text>
                <Select data={LLMs} value={llm}
                        onChange={(value) => {
                            setLlm(value!)
                        }} allowDeselect={false}/>

                <Text mt='md'>Temperature</Text>
                <Slider label={temperature} min={0} max={1} step={0.01} value={temperature} onChange={setTemperature}
                        disabled={llm == 'none'}/>

                <Group justify={'flex-end'} mt={'md'}>
                    <Button style={{
                        background: 'var(--mantine-color-red-9)',
                        hover: 'var(--mantine-color-red-8)',
                        color: 'var(--mantine-color-white)',
                        border: 'none',
                    }}
                            onClick={() => {
                                setLlm('gpt-3.5-turbo');
                                setTemperature(0.5)
                            }}
                    >
                        Reset
                    </Button>
                    <Button variant="gradient"
                            gradient={{from: 'blue', to: 'cyan', deg: 150}}
                            onClick={() => {
                                setShowLlmModal(false);
                                if (stepper < 5) setStepper(5);
                                setStepperDescription5(`Model: ${llm}|Temperature: ${temperature}`)
                                notifications.show({
                                    color: 'teal',
                                    title: 'Success!',
                                    message: 'LLM configuration set successfully',
                                    icon: <IconCheck style={{width: rem(18), height: rem(18)}}/>,
                                    loading: false,
                                    withCloseButton: true,
                                    autoClose: 5000,
                                })
                            }}
                    >
                        Submit
                    </Button>
                </Group>
            </Modal>
        </div>
    );
}

export function useContext() {
    return useOutletContext<ContextType>();
}
