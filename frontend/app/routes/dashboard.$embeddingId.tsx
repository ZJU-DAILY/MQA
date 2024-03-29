import {LinksFunction} from "@remix-run/node";
import {
    ActionIcon,
    Avatar,
    Center,
    Container,
    Flex,
    Group,
    Image,
    Popover,
    rem,
    ScrollArea,
    SimpleGrid,
    Skeleton,
    Stack,
    Text,
    Textarea,
    useMantineTheme,
} from "@mantine/core";
import {useEffect, useRef, useState} from "react";
import {IconArrowUp, IconCloudUpload, IconDownload, IconPhoto, IconX} from "@tabler/icons-react";
import {white} from "kleur/colors";
import {useContext} from "~/routes/dashboard";
import {Dropzone, FileWithPath, IMAGE_MIME_TYPE} from "@mantine/dropzone";
import styles from "~/styles/DashboardMain.css";
import {notifications} from "@mantine/notifications";
import bgL from "~/assets/bg-left.png"
import bgR from "~/assets/bg-right.png"

export const links: LinksFunction = () => [
    {rel: "stylesheet", href: styles},
];

type User = {
    name: string;
    avatar: string;
}

type Img = {
    id: string,
}

type Message = {
    user: User,
    images: Img[],
    chosen: number,
    text: string,
}

export default function DashboardMain() {
    // for mantine color render
    const theme = useMantineTheme();

    // const {embeddingId} = useLoaderData<typeof loader>();
    const {
        stepper,
        useKnowledge,
        algorithm,
        neighbor,
        candidate,
        indexWeight,
        retrievalFramework,
        retrievalWeight,
        llm,
        temperature,
        resultNumber
    } = useContext();
    const [loading, setLoading] = useState(false);

    // chat parameters
    const [messages, setMessages] = useState<Message[]>([]);
    const system: User = {name: 'MQA', avatar: ''};
    const user: User = {name: 'user', avatar: ''}

    // user inputs
    // target modal
    const [selectedTarget, setSelectedTarget] = useState<number>(-1);
    // text
    const [inputText, setInputText] = useState<string>('');
    // image
    const [inputImage, setInputImage] = useState<FileWithPath | null>(null);
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const openRef = useRef<() => void>(null);

    useEffect(() => {
        if (inputImage == null) return;
        // if (imageUrl) URL.revokeObjectURL(imageUrl);
        setImageUrl(URL.createObjectURL(inputImage));
    }, [inputImage]);

    // use to auto scroll to bottom
    const scrollAreaRef = useRef<HTMLDivElement>(null);

    const handleSubmit = async () => {
        if (inputText.trim() === '' && inputImage == null) return;
        if (stepper < 2 && useKnowledge) {
            notifications.show({
                color: 'red',
                icon: <IconX style={{width: rem(18), height: rem(18)}}/>,
                autoClose: 5000,
                withCloseButton: true,
                title: 'Search Error',
                message: 'Please process `Embedding Configuration` first',
            });
            return;
        }
        setLoading(true);
        setSelectedTarget(-1);

        const newMessages = [...messages];
        // input user message
        const userMessage = {
            user: user,
            images: imageUrl ? [{id: imageUrl}] : [],
            chosen: -1,
            text: inputText || `[only image]`,
        }
        newMessages.push(userMessage);
        setMessages(newMessages);

        // input placeholder message for skeleton
        const placeholderMessage: Message = {
            user: system,
            images: [{id: '-2'}, {id: '-3'}, {id: '-4'}],
            chosen: -1,
            text: "",
        }
        newMessages.push(placeholderMessage)
        setMessages(newMessages);

        // scroll to bottom
        scrollAreaRef.current!.scrollTo({top: scrollAreaRef.current!.scrollHeight, behavior: 'smooth'});

        // search from backend
        const formData = new FormData();
        formData.append('useKnowledge', useKnowledge.toString());
        // formData.append('embeddingId', embeddingId == undefined || !useKnowledge ? '-1' : embeddingId);
        formData.append('selectedTarget', selectedTarget.toString());
        formData.append('text', inputText);
        inputImage && formData.append('image', inputImage);
        formData.append('algorithm', algorithm);
        formData.append('neighbor', neighbor.toString());
        formData.append('candidate', candidate.toString());
        formData.append('indexWeight', JSON.stringify(indexWeight));
        formData.append('retrievalFramework', retrievalFramework);
        formData.append('retrievalWeight', JSON.stringify(retrievalWeight));
        formData.append('llm', llm);
        formData.append('temperature', temperature.toString());
        formData.append('resultNumber', resultNumber.toString());
        setInputText('');
        setInputImage(null);
        setImageUrl(null);
        
        const response = await fetch(`http://127.0.0.1:4523/m1/4132394-0-default/search`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        console.log(data);
        if (data.message) {
            newMessages[newMessages.length - 1] = {
                user: system,
                images: [],
                chosen: -1,
                text: data.message,
            };
            setMessages(newMessages);
        } else {
            newMessages[newMessages.length - 1] = {
                user: system,
                images: data.data.images,
                chosen: -1,
                text: data.data.reply,
            };
            setMessages(newMessages);
        }

        // scroll to bottom again
        scrollAreaRef.current!.scrollTo({top: scrollAreaRef.current!.scrollHeight, behavior: 'smooth'});
        setLoading(false);
    }

    return (
        <Container
            style={{
                height: '90vh',
                justifyContent: 'space-between',
                alignContent: 'center',
                display: 'flex',
                flexDirection: 'column'
            }}
            w={'auto'}
        >
            <ScrollArea viewportRef={scrollAreaRef} style={{height: 'calc(90vh - 50px)'}}>
                {messages.length == 0 ?
                    (
                        <Stack align={'center'} mt={'calc((100vh - 650px) / 2)'}>
                            <Text
                                size={rem(50)}
                                fw={900}
                                style={{fontFamily: 'Arial'}}
                            >MQA</Text>
                            <Text size={'xl'}>Interactive, Pluggable, and Multi-modal Query Answering System</Text>
                            <Text c={'dimmed'}>Powered by MUST and LLMs</Text>
                            <SimpleGrid cols={2} className={"hero-title-image"}>
                                <Center><Image src={bgL}/></Center>
                                <Center><Image src={bgR}/></Center>
                            </SimpleGrid>
                        </Stack>
                    ) :
                    messages.map((message, index) =>
                        <Flex key={index} justify="flex-start" align="flex-start" gap="md" mb={'md'}>
                            {message.user.avatar
                                ? <Avatar variant={'light'} alt={message.user.name} mt={7} src={message.user.avatar}/>
                                : <Avatar variant={'light'} alt={message.user.name}
                                          mt={7}>{message.user.name[0].toUpperCase()}</Avatar>}
                            <Stack justify="flex-start" gap="xs" style={{width: '90%'}}>
                                <Text mt={10} fw={700} size="xl">{message.user.name}</Text>
                                {message.text
                                    ? (<Text size={rem(20)}>{message.text}</Text>)
                                    : (<Skeleton height={rem(20)} width='100%'></Skeleton>)}
                                <SimpleGrid cols={3} spacing={'xs'}>
                                    {message.images?.map((image, _index) => (
                                            <Container
                                                key={_index}
                                                style={{
                                                    borderRadius: 8,
                                                    border: message.chosen === _index ? '2px solid #2f9e44' : '2px solid #F0F0F0',
                                                    display: 'inline-block',
                                                    padding: 8
                                                }}
                                                onClick={() => {
                                                    if (message.images[0].id.startsWith('-') || llm == 'dall-e-3') {
                                                        return;
                                                    }
                                                    const target = selectedTarget == _index ? -1 : _index;
                                                    message.chosen = target;
                                                    setSelectedTarget(target);
                                                }}
                                            >
                                                {image.id.startsWith('-')
                                                    ? <>
                                                        <Skeleton height='200' w={'200'}/>
                                                        <Center>
                                                            <Skeleton height='16' width='40%' mt='5'/>
                                                        </Center>
                                                    </> :
                                                    <Image
                                                        src={image.id}
                                                        fallbackSrc="https://placehold.co/300x200?text=Placeholder"
                                                        // alt={image.id}
                                                    />
                                                }
                                            </Container>
                                        )
                                    )}
                                </SimpleGrid>
                                {
                                    message.user.name == 'MQA' && message.images?.length > 0 && !message.images[0].id.startsWith('-') && llm != 'dall-e-3' &&
                                    <Center>
                                        <Text size={'sm'} c="dimmed">You can choose your favorite image by clicking
                                            it</Text>
                                    </Center>
                                }
                            </Stack>
                        </Flex>
                    )
                }
            </ScrollArea>
            <Center w={'100%'}>
                <Textarea
                    bottom={0}
                    maw={800}
                    w={'100%'}
                    radius="lg"
                    size="lg"
                    placeholder="Type your message..."
                    rightSectionWidth={100}
                    minRows={1}
                    maxRows={8}
                    value={inputText}
                    onChange={(event) => {
                        setInputText(event.target.value);
                    }}
                    onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                            event.preventDefault();
                            handleSubmit();
                        }
                    }}
                    autosize
                    rightSection={
                        <Group justify={'flex-end'}>
                            <Popover>
                                <Popover.Target>
                                    <IconPhoto
                                        size={32}
                                        radius="xl"
                                        color={theme.colors.gray[4]}
                                        stroke={1.5}
                                        className={'icon'}
                                    />
                                </Popover.Target>
                                <Popover.Dropdown>
                                    <div className={"wrapper"}>
                                        {inputImage && <Image height={200} w="auto" src={imageUrl}/>}
                                        <Dropzone
                                            openRef={openRef}
                                            onDrop={(files) => {
                                                setInputImage(files[0])
                                            }}
                                            className={'dropzone'}
                                            radius="md"
                                            accept={IMAGE_MIME_TYPE}
                                            maxSize={30 * 1024 ** 2}>
                                            <div style={{pointerEvents: 'none'}}>
                                                <Group justify="center">
                                                    <Dropzone.Accept>
                                                        <IconDownload
                                                            style={{width: rem(50), height: rem(50)}}
                                                            color={theme.colors.blue[6]}
                                                            stroke={1.5}
                                                        />
                                                    </Dropzone.Accept>
                                                    <Dropzone.Reject>
                                                        <IconX
                                                            style={{width: rem(50), height: rem(50)}}
                                                            color={theme.colors.red[6]}
                                                            stroke={1.5}
                                                        />
                                                    </Dropzone.Reject>
                                                    <Dropzone.Idle>
                                                        <IconCloudUpload style={{width: rem(50), height: rem(50)}}
                                                                         stroke={1.5}/>
                                                    </Dropzone.Idle>
                                                </Group>

                                                <Text ta="center" fw={700} fz="lg" mt="xl">
                                                    <Dropzone.Accept>Drop files here</Dropzone.Accept>
                                                    <Dropzone.Reject>Image less than 30mb</Dropzone.Reject>
                                                    <Dropzone.Idle>Upload resume</Dropzone.Idle>
                                                </Text>
                                                <Text ta="center" fz="sm" mt="xs" c="dimmed">
                                                    Drag&apos;n&apos;drop files here to upload. We can accept
                                                    only <i>Image</i> files that
                                                    are less than 30mb in size.
                                                </Text>
                                            </div>
                                        </Dropzone>
                                    </div>
                                </Popover.Dropdown>
                            </Popover>

                            <ActionIcon disabled={!inputText.trim() && inputImage == null}
                                        onClick={() => {
                                            handleSubmit();
                                        }}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter') {
                                                event.preventDefault();
                                                handleSubmit();
                                            }
                                        }}
                                        size={32}
                                        radius="xl"
                                        loading={loading}
                                        color={white()} variant="filled">
                                <IconArrowUp style={{width: rem(18), height: rem(18)}} stroke={1.5}/>
                            </ActionIcon>
                        </Group>
                    }
                />
            </Center>
        </Container>
    )
}